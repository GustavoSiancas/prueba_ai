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
)
from app.infrastructure.nlp.align_judge import comparar_descripcion_con_resumen_ia
from app.infrastructure.redisdb.rate_limit import BudgetGuardrails
from app.infrastructure.redisdb.indices import recent_candidates, get_features_by_url, save_video_features as redis_save_features
from app.infrastructure.pg.dao import pg_get_by_url, pg_save_video_features
from app.infrastructure.audio.ffmpeg import extract_wav_mono16k
from app.infrastructure.audio.transcribe import transcribe_audio
from app.infrastructure.nlp.vlm_summary import _uniform_keyframes

@dataclass
class EvaluateService:
    """
    Servicio de negocio para la evaluación de videos.

    Pipeline (una sola descarga del base si hace falta):
      1) Cache lookup: intenta huellas en Redis (bits) y, si MISS, fallback a PG y repoblado de Redis.
      2) Deduplicación rápida:
         - pHash64 "mayoritario" (64 bits) para descarte grueso (HASH gate).
         - Huella de secuencia (bool[M,64]) con tolerancia temporal (SEQ gate).
      3) Si NO es duplicado:
         - Obtiene keyframes del propio MP4 **en memoria** (efímeros) cuando haga falta.
         - Audio opcional: extrae WAV 16 kHz mono y pide ASR (Whisper) si `AUDIO_ASR_ENABLED=true`.
         - VLM: genera un resumen usando **frames base64** (no requiere .mp4) + transcripción si la hay.
         - Juez de alineación: compara el resumen con `descripcion` y devuelve `AlignmentResult`.
      4) Persistencia:
         - Redis: `phash64_b` (8 bytes) y `seq_b` (N*8 bytes) + metadatos.
         - Postgres (si `PG_ENABLED=true`): write-through de las mismas huellas.
      5) Costos (opcional): si `BUDGET_GUARDRAILS_ENABLED=true`, evalúa/commitea contadores.

    Garantías:
      - La descarga del video base ocurre como máximo una vez por request (sólo si faltan insumos).
      - Los keyframes y el audio son **temporales** y se eliminan al final de la request.
    """

    settings: Settings

    def _mktemp(self):
        """
        Crea un directorio temporal aislado por request para archivos locales.

        Estructura:
            <tmp>/frames/ -> (no persistimos aquí frames; sólo si hiciera falta)
            <tmp>/audio/  -> WAV temporal para ASR

        Limpieza:
            Siempre se borra en el `finally` de evaluate().
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
        Nota: no modifica estado externo si el video resulta duplicado.
        """
        # --- 0) Budget guardrails (opcional) ---------------------------------
        guard = None
        if getattr(self.settings, "BUDGET_GUARDRAILS_ENABLED", False):
            guard = BudgetGuardrails(
                usd_day_cap=self.settings.USD_DAY_CAP,
                max_llm_calls=self.settings.MAX_LLM_CALLS,
                max_emb_calls=self.settings.MAX_EMB_CALLS
            )

        root, frames_dir, audio_dir = self._mktemp()
        try:
            # --- 1) Lookup de huellas (Redis -> PG fallback) ---------------------
            cached_base = get_features_by_url(req.campaign_id, str(req.video_url))
            if not cached_base and getattr(self.settings, "PG_ENABLED", False):
                rec = pg_get_by_url(str(req.video_url))
                if rec:
                    try:
                        redis_save_features(
                            campaign_id=req.campaign_id,
                            video_url=rec["url"],
                            phash64_bits=rec["phash64"],
                            seq_bits=rec["seq_sig"],
                            duration_s=rec["duration_s"],
                        )
                    except Exception as e:
                        print(f"[warn] no se pudo repoblar Redis desde PG: {e}")
                    cached_base = rec

            if cached_base:
                return EvaluateResponse(
                    duplicated=True,
                    duplicate_reason="URL",
                    duplicate_candidate_url=str(req.video_url),
                    alignment=None,
                    cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                )

            base_fp = None
            base_seq = None
            base_path = None
            frames_b64 = []
            transcript_text = None

            if cached_base:
                base_fp = np.array(cached_base["phash64"], dtype=np.uint8)
                base_seq = np.array(cached_base["seq_sig"], dtype=bool)

            # --- 1.a) Dedupe contra recientes (si ya tengo huellas del base) -----
            if base_fp is not None and base_seq is not None:
                for cand in recent_candidates(req.campaign_id, k=50):
                    cand_fp = np.array(cand["phash64"], dtype=np.uint8)
                    if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="HASH",
                            duplicate_candidate_url=cand["url"],
                            alignment=None,
                            cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                        )
                    cand_seq = np.array(cand["seq_sig"], dtype=bool)
                    if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="SEQ",
                            duplicate_candidate_url=cand["url"],
                            alignment=None,
                            cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                        )

            # --- 1.b) Dedupe contra candidates explícitos ------------------------
            for cand_url in req.candidates:
                if str(cand_url) == str(req.video_url):
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="URL",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                    )

                cached_cand = get_features_by_url(req.campaign_id, str(cand_url))
                if cached_cand:
                    if base_fp is None or base_seq is None:
                        # DESCARGA ÚNICA AQUÍ si faltan huellas
                        base_path = base_path or descargar_video(
                            str(req.video_url), output_folder=root,
                            size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                        )
                        if not base_path:
                            raise RuntimeError("No se pudo descargar el video base.")
                        base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
                        base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

                        # EXTRAER KEYFRAMES (EFÍMEROS) + AUDIO opcional en la misma pasada
                        frames_b64 = _uniform_keyframes(base_path, max_frames=self.settings.FRAMES_MAX)
                        if getattr(self.settings, "AUDIO_ASR_ENABLED", True):
                            wav_path = os.path.join(audio_dir, "audio.wav")
                            if extract_wav_mono16k(base_path, wav_path, sr=self.settings.AUDIO_TARGET_SR):
                                transcript_text = transcribe_audio(wav_path)

                    cand_fp = np.array(cached_cand["phash64"], dtype=np.uint8)
                    if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="HASH",
                            duplicate_candidate_url=cached_cand["url"],
                            alignment=None,
                            cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                        )
                    cand_seq = np.array(cached_cand["seq_sig"], dtype=bool)
                    if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="SEQ",
                            duplicate_candidate_url=cached_cand["url"],
                            alignment=None,
                            cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                        )
                    continue

                # Candidate sin cache → descarga candidate (base se descarga una sola vez si faltan huellas)
                if base_fp is None or base_seq is None:
                    base_path = base_path or descargar_video(
                        str(req.video_url), output_folder=root,
                        size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                    )
                    if not base_path:
                        raise RuntimeError("No se pudo descargar el video base.")
                    base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
                    base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

                    # EXTRAER KEYFRAMES (EFÍMEROS) + AUDIO opcional en la misma pasada
                    frames_b64 = _uniform_keyframes(base_path, max_frames=self.settings.FRAMES_MAX)
                    if getattr(self.settings, "AUDIO_ASR_ENABLED", True):
                        wav_path = os.path.join(audio_dir, "audio.wav")
                        if extract_wav_mono16k(base_path, wav_path, sr=self.settings.AUDIO_TARGET_SR):
                            transcript_text = transcribe_audio(wav_path)

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
                        cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                    )
                cand_seq = frame_hash_sequence(cand_path, seconds_interval=2.0, max_frames=60, hash_size=8)
                if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="SEQ",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls":0,"embedding_calls":0,"transcription_seconds":0,"degraded_path":False}
                    )

            # --- 2) Budget decision (opcional) -----------------------------------
            degraded = False
            if guard is not None:
                decision = guard.decide(req.campaign_id, est_usd=0.02, need={"llm": 1, "emb": 0})
                degraded = (decision != "allow")
            if degraded:
                return EvaluateResponse(
                    duplicated=False,
                    duplicate_reason=None,
                    duplicate_candidate_url=None,
                    alignment=AlignmentResult(
                        aproved=False, match_percent=0.0,
                        reasons="Evaluación semántica degradada por presupuesto. Reintenta más tarde."
                    ),
                    cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": True}
                )

            # --- 3) Preparar insumos para VLM sin re-descargar -------------------
            # Si aún no tenemos frames (porque nunca descargamos base antes), descárgalo ahora
            if not frames_b64:
                if base_path is None:
                    base_path = descargar_video(
                        str(req.video_url), output_folder=root,
                        size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                    )
                    if not base_path:
                        raise RuntimeError("No se pudo descargar el video base.")
                # Generar keyframes EFÍMEROS
                frames_b64 = _uniform_keyframes(base_path, max_frames=self.settings.FRAMES_MAX)

                # Extraer audio/ASR opcional
                if getattr(self.settings, "AUDIO_ASR_ENABLED", True) and transcript_text is None:
                    wav_path = os.path.join(audio_dir, "audio.wav")
                    if extract_wav_mono16k(base_path, wav_path, sr=self.settings.AUDIO_TARGET_SR):
                        transcript_text = transcribe_audio(wav_path)

            # --- 4) Resumen VLM + juez de alineación -----------------------------
            summary = analyze_frames_free_narrative(frames_b64, transcript_text=transcript_text)

            llm_calls = 1 if summary else 0
            if not summary:
                return EvaluateResponse(
                    duplicated=False,
                    duplicate_reason=None,
                    duplicate_candidate_url=None,
                    alignment=AlignmentResult(
                        aproved=False,
                        match_percent=0.0,
                        reasons="No se pudo generar el resumen del video."),
                    cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                )

            _compact_text = summarize_video_textual(summary)

            cmp_raw = comparar_descripcion_con_resumen_ia(req.descripcion, summary, umbral_aprobacion=70)
            if not cmp_raw:
                return EvaluateResponse(
                    duplicated=False,
                    duplicate_reason=None,
                    duplicate_candidate_url=None,
                    alignment=AlignmentResult(
                        aproved=False,
                        match_percent=0.0,
                        reasons="No se pudo comparar la descripción con el resumen."),
                    cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                )
            try:
                cmp_json = json.loads(cmp_raw)
            except Exception:
                cmp_json = {"aproved": False, "match_percent": 0.0, "reasons": "Respuesta IA inválida."}

            # --- 5) Commit presupuesto (si aplica) --------------------------------
            if guard is not None and llm_calls:
                guard.commit(req.campaign_id, spent_usd=0.02, used={"llm": llm_calls, "emb": 0})

            # --- 6) Persistencia SOLO si el video fue APROBADO --------------------
            is_approved = bool(cmp_json.get("aproved", False))

            # Guardar huellas únicamente si:
            #   - fue aprobado por el VLM
            #   - y no estaban ya cacheadas (primer ingreso)
            if is_approved and not cached_base:
                # Asegura que tenemos huellas; en el camino sin candidates podrían seguir en None
                if base_fp is None or base_seq is None:
                    if base_path is None:
                        base_path = descargar_video(
                            str(req.video_url), output_folder=root,
                            size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                        )
                        if not base_path:
                            raise RuntimeError("No se pudo descargar el video base para persistir huellas.")
                    base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
                    base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

                if base_fp is not None and base_seq is not None:
                    # Redis
                    video_id = None
                    try:
                        video_id = redis_save_features(
                            campaign_id=req.campaign_id,
                            video_url=str(req.video_url),
                            phash64_bits=base_fp,
                            seq_bits=base_seq,
                            duration_s=get_duration_s(base_path) if base_path else 0.0,
                        )
                    except Exception as e:
                        print(f"[warn] no se pudieron guardar huellas en Redis: {e}")

                    # Postgres write-through (solo si está habilitado)
                    if getattr(self.settings, "PG_ENABLED", False):
                        try:
                            if not video_id:
                                video_id = hashlib.sha1(str(req.video_url).encode()).hexdigest()
                            pg_save_video_features(
                                video_id=video_id,
                                campaign_id=req.campaign_id,
                                url=str(req.video_url),
                                phash64_bits=base_fp,
                                seq_bits=base_seq,
                                duration_s=get_duration_s(base_path) if base_path else 0.0,
                            )
                        except Exception as e:
                            print(f"[warn] no se pudieron guardar huellas en Postgres: {e}")
            else:
                pass

            return EvaluateResponse(
                duplicated=False, duplicate_reason=None, duplicate_candidate_url=None,
                alignment=AlignmentResult(
                    aproved=bool(cmp_json.get("aproved", False)),
                    match_percent=float(cmp_json.get("match_percent", 0.0)),
                    reasons=str(cmp_json.get("reasons", "Sin motivos."))
                ),
                cost={"llm_calls": llm_calls, "embedding_calls": 0,
                      "transcription_seconds": 0, "degraded_path": False}
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)