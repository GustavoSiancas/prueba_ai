import os, json, tempfile, shutil, hashlib
import numpy as np
from dataclasses import dataclass

from app.api.http.schemas.requests import EvaluateRequest
from app.api.http.schemas.responses import EvaluateResponse, AlignmentResult
from app.infrastructure.settings import Settings
from app.infrastructure.downloading.downloader import descargar_video
from app.infrastructure.cv.phash import video_fingerprint, similarity_percent
from app.infrastructure.cv.sequence import frame_hash_sequence, sequence_match_percent, get_duration_s
from app.infrastructure.nlp.vlm_summary import (
    analyze_frames_free_narrative,
    summarize_video_textual,
    _uniform_keyframes,
)
from app.infrastructure.nlp.align_judge import comparar_descripcion_con_resumen_ia
from app.infrastructure.pg.dao import pg_get_by_url, pg_save_video_features, pg_recent_candidates
from app.infrastructure.audio.ffmpeg import extract_wav_mono16k
from app.infrastructure.audio.transcribe import transcribe_audio


@dataclass
class EvaluateService:
    """
    Servicio de negocio para la evaluación de videos (PG-only).

    Pipeline (una sola descarga del base si hace falta):
      1) Lookup: intenta en Postgres por URL. Si EXISTE -> short-circuit (duplicado por URL).
      2) Deduplicación rápida:
         - pHash64 (64 bits) para descarte grueso (HASH gate).
         - Secuencia de dHash por frame (bool[M,64]) con tolerancia temporal (SEQ gate).
      3) Si NO es duplicado:
         - Extrae keyframes del MP4 (en memoria, EFÍMEROS) y audio opcional (ASR).
         - VLM con frames base64 + transcript opcional.
         - Juez de alineación contra `descripcion`.
      4) Persistencia SOLO si aprueba:
         - Guarda huellas en Postgres (source of truth).
      5) Costos: (desactivado aquí; reintroducir si se necesita).

    Garantías:
      - El MP4 del base se descarga como máximo una vez por request (si faltan insumos).
      - Keyframes/audio son EFÍMEROS y se eliminan al final de la request.
    """

    settings: Settings

    def _mktemp(self):
        """
        Directorio temporal aislado por request.

        Estructura:
          <tmp>/frames/ -> (no persistimos aquí frames; sólo si hiciera falta)
          <tmp>/audio/  -> WAV temporal para ASR
        """
        root = tempfile.mkdtemp(prefix="req_")
        frames = os.path.join(root, "frames")
        audio = os.path.join(root, "audio")
        os.makedirs(frames, exist_ok=True)
        os.makedirs(audio, exist_ok=True)
        return root, frames, audio

    def evaluate(self, req: EvaluateRequest) -> EvaluateResponse:
        """
        Orquesta el flujo completo de dedupe + alineación.
        No modifica estado si el video resulta duplicado.
        """

        root, frames_dir, audio_dir = self._mktemp()
        try:
            # --- 1) Lookup por URL en PG (short-circuit duplicado)
            cached = pg_get_by_url(str(req.video_url))
            if cached:
                return EvaluateResponse(
                    duplicated=True,
                    duplicate_reason="URL",
                    duplicate_candidate_url=str(req.video_url),
                    alignment=None,
                    cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                )

            # --- 2) Descarga base UNA sola vez y calcula huellas/insumos
            base_path = descargar_video(
                str(req.video_url),
                output_folder=root,
                size_mb_limit=self.settings.VIDEO_MAX_MB,
                timeout_s=self.settings.DL_TIMEOUT_S,
            )
            if not base_path:
                raise RuntimeError("No se pudo descargar el video base.")

            # Huellas para dedupe
            base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
            base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

            # Keyframes EFÍMEROS + ASR opcional para VLM
            frames_b64 = _uniform_keyframes(base_path, max_frames=self.settings.FRAMES_MAX)
            transcript_text = None
            if getattr(self.settings, "AUDIO_ASR_ENABLED", True):
                wav_path = os.path.join(audio_dir, "audio.wav")
                if extract_wav_mono16k(base_path, wav_path, sr=self.settings.AUDIO_TARGET_SR):
                    transcript_text = transcribe_audio(wav_path)

            # --- 3) Dedup contra recientes (PG)
            if base_fp is not None and base_seq is not None:
                for cand in pg_recent_candidates(req.campaign_id, k=50):
                    cand_fp = np.array(cand["phash64"], dtype=np.uint8)
                    if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="HASH",
                            duplicate_candidate_url=cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                        )
                    cand_seq = np.array(cand["seq_sig"], dtype=bool)
                    if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="SEQ",
                            duplicate_candidate_url=cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                        )

            # --- 4) Dedup contra candidates explícitos (PG)
            for cand_url in req.candidates:
                # Igual URL -> duplicado directo
                if str(cand_url) == str(req.video_url):
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="URL",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                    )

                cached_cand = pg_get_by_url(str(cand_url))
                if cached_cand:
                    cand_fp = np.array(cached_cand["phash64"], dtype=np.uint8)
                    if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="HASH",
                            duplicate_candidate_url=cached_cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                        )
                    cand_seq = np.array(cached_cand["seq_sig"], dtype=bool)
                    if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="SEQ",
                            duplicate_candidate_url=cached_cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                        )
                    continue

                # Candidate sin features -> descargar candidate y comparar (barato)
                cand_path = descargar_video(
                    str(cand_url), output_folder=root,
                    size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                )
                if not cand_path:
                    continue
                cand_fp = video_fingerprint(cand_path, seconds_interval=5.0, max_frames=20)
                if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="HASH",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                    )
                cand_seq = frame_hash_sequence(cand_path, seconds_interval=2.0, max_frames=60, hash_size=8)
                if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="SEQ",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                    )

            # --- 5) VLM + juez de alineación
            summary = analyze_frames_free_narrative(frames_b64, transcript_text=transcript_text)
            llm_calls = 1 if summary else 0
            if not summary:
                return EvaluateResponse(
                    duplicated=False,
                    duplicate_reason=None,
                    duplicate_candidate_url=None,
                    alignment=AlignmentResult(
                        aproved=False, match_percent=0.0,
                        reasons="No se pudo generar el resumen del video."
                    ),
                    cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                )

            _compact_text = summarize_video_textual(summary)

            cmp_raw = comparar_descripcion_con_resumen_ia(req.descripcion, summary, umbral_aprobacion=70)
            if not cmp_raw:
                return EvaluateResponse(
                    duplicated=False,
                    duplicate_reason=None,
                    duplicate_candidate_url=None,
                    alignment=AlignmentResult(
                        aproved=False, match_percent=0.0,
                        reasons="No se pudo comparar la descripción con el resumen."
                    ),
                    cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
                )
            try:
                cmp_json = json.loads(cmp_raw)
            except Exception:
                cmp_json = {"aproved": False, "match_percent": 0.0, "reasons": "Respuesta IA inválida."}

            # --- 6) Persistencia SOLO si aprueba (no guardamos rechazados)
            is_approved = bool(cmp_json.get("aproved", False))
            if is_approved:
                video_id = hashlib.sha1(str(req.video_url).encode()).hexdigest()
                pg_save_video_features(
                    video_id=video_id,
                    campaign_id=req.campaign_id,
                    url=str(req.video_url),
                    phash64_bits=base_fp,
                    seq_bits=base_seq,
                    duration_s=get_duration_s(base_path) if base_path else 0.0,
                )

            # --- 7) Respuesta final
            return EvaluateResponse(
                duplicated=False,
                duplicate_reason=None,
                duplicate_candidate_url=None,
                alignment=AlignmentResult(
                    aproved=is_approved,
                    match_percent=float(cmp_json.get("match_percent", 0.0)),
                    reasons=str(cmp_json.get("reasons", "Sin motivos.")),
                ),
                cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False},
            )

        finally:
            shutil.rmtree(root, ignore_errors=True)
