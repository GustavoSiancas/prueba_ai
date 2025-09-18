import os, json, tempfile, shutil
import numpy as np
from dataclasses import dataclass

from app.api.http.schemas.requests import EvaluateRequest
from app.api.http.schemas.responses import EvaluateResponse, AlignmentResult
from app.infrastructure.settings import Settings
from app.infrastructure.downloading.downloader import descargar_video
from app.infrastructure.cv.phash import video_fingerprint, similarity_percent
from app.infrastructure.cv.sequence import frame_hash_sequence, sequence_match_percent, get_duration_s
from app.infrastructure.nlp.vlm_summary import (
    analyze_video_free_narrative,
    analyze_video_hybrid,
    summarize_video_textual,
)
from app.infrastructure.nlp.align_judge import comparar_descripcion_con_resumen_ia
from app.infrastructure.redisdb.rate_limit import BudgetGuardrails
from app.infrastructure.redisdb.indices import recent_candidates, get_features_by_url, save_video_features


@dataclass
class EvaluateService:
    """
    Servicio de negocio para la evaluación de videos.

    Responsabilidades:
        - Detección de duplicados (pHash/seq) usando Redis como caché de huellas.
        - Control de gasto/llamadas (BudgetGuardrails).
        - Resumen VLM (free o hybrid) y juicio de alineación.
        - Persistencia de huellas del video base si no existían.
    """

    settings: Settings

    def _mktemp(self):
        """
        Crea un directorio temporal aislado por request para archivos locales.
        Devuelve: (root, frames_dir, audio_dir)

        Limpieza:
        - Se elimina en `finally` de evaluate().
        """

        root = tempfile.mkdtemp(prefix="req_")
        frames = os.path.join(root, "frames")
        audio = os.path.join(root, "audio")
        os.makedirs(frames, exist_ok=True)
        os.makedirs(audio, exist_ok=True)
        return root, frames, audio

    def evaluate(self, req: EvaluateRequest) -> EvaluateResponse:
        """
        Orquestación principal:
            1) Intenta usar huellas desde Redis (cache HIT). Si hay, compara contra recientes y candidates sin recalcular.
            2) Si falta algo, descarga y calcula pHash/seq del base/candidates.
            3) Si no hay duplicado, decide presupuesto y, si procede, llama al VLM para resumir y luego juzga alineación.
            4) Guarda huellas del base en Redis si no existían.

        Retorna:
            EvaluateResponse listo para el cliente.
        """

        guard = BudgetGuardrails(
            usd_day_cap=self.settings.USD_DAY_CAP,
            max_llm_calls=self.settings.MAX_LLM_CALLS,
            max_emb_calls=self.settings.MAX_EMB_CALLS
        )

        root, frames_dir, audio_dir = self._mktemp()
        try:
            # 1) Intenta cache de huellas del base
            cached_base = get_features_by_url(req.campaign_id, str(req.video_url))
            base_fp = None
            base_seq = None
            base_path = None # solo se descarga si se necesita

            if cached_base:
                base_fp = np.array(cached_base["phash64"], dtype=np.uint8)
                base_seq = np.array(cached_base["seq_sig"], dtype=bool)

            # 1.a) Dedupe contra recientes (solo si ya hay huellas base)
            if base_fp is not None and base_seq is not None:
                for cand in recent_candidates(req.campaign_id, k=50):
                    # HASH gate
                    cand_fp = np.array(cand["phash64"], dtype=np.uint8)
                    if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="HASH",
                            duplicate_candidate_url=cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                        )
                    # SEQ gate
                    cand_seq = np.array(cand["seq_sig"], dtype=bool)
                    if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="SEQ",
                            duplicate_candidate_url=cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                        )

            # 1.b) Dedupe contra candidates explícitos
            for cand_url in req.candidates:
                # Atajo: si candidate == base_url, marcamos como duplicado directo
                if str(cand_url) == str(req.video_url):
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="HASH",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                    )

                cached_cand = get_features_by_url(req.campaign_id, str(cand_url))
                if cached_cand:
                    # Si el base no tenía huellas en cache, calcúlalas (una sola vez)
                    if base_fp is None or base_seq is None:
                        base_path = base_path or descargar_video(
                            str(req.video_url), output_folder=root,
                            size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                        )
                        if not base_path:
                            raise RuntimeError("No se pudo descargar el video base.")
                        base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
                        base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

                    # Compara contra huellas del candidate desde Redis
                    cand_fp = np.array(cached_cand["phash64"], dtype=np.uint8)
                    if similarity_percent(base_fp, cand_fp) >= self.settings.HASH_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="HASH",
                            duplicate_candidate_url=cached_cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                        )
                    cand_seq = np.array(cached_cand["seq_sig"], dtype=bool)
                    if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                        return EvaluateResponse(
                            duplicated=True,
                            duplicate_reason="SEQ",
                            duplicate_candidate_url=cached_cand["url"],
                            alignment=None,
                            cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                        )
                    continue # siguiente candidate

                # Candidate sin cache: descarga + calcula huellas baratas
                if base_fp is None or base_seq is None:
                    base_path = base_path or descargar_video(
                        str(req.video_url), output_folder=root,
                        size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                    )
                    if not base_path:
                        raise RuntimeError("No se pudo descargar el video base.")
                    base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
                    base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

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
                        cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                    )
                cand_seq = frame_hash_sequence(cand_path, seconds_interval=2.0, max_frames=60, hash_size=8)
                if sequence_match_percent(base_seq, cand_seq, bit_tolerance=5, window=2) >= self.settings.SEQ_DUP_THRESHOLD:
                    return EvaluateResponse(
                        duplicated=True,
                        duplicate_reason="SEQ",
                        duplicate_candidate_url=str(cand_url),
                        alignment=None,
                        cost={"llm_calls": 0, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                    )

            # 2) Presupuesto y posible degradación
            decision = guard.decide(req.campaign_id, est_usd=0.02, need={"llm": 1, "emb": 0})
            degraded = (decision != "allow")
            if degraded:
                # Camino barato sin IA cara
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

            # 3) Descargar base si aún no se descargó y preparar huellas si faltaran
            if base_path is None:
                base_path = descargar_video(
                    str(req.video_url), output_folder=root,
                    size_mb_limit=self.settings.VIDEO_MAX_MB, timeout_s=self.settings.DL_TIMEOUT_S
                )
                if not base_path:
                    raise RuntimeError("No se pudo descargar el video base.")
                if base_fp is None or base_seq is None:
                    base_fp = video_fingerprint(base_path, seconds_interval=5.0, max_frames=20)
                    base_seq = frame_hash_sequence(base_path, seconds_interval=2.0, max_frames=60, hash_size=8)

            # 3.a) Resumen VLM (free/hybrid)
            mode = (self.settings.SUMMARY_MODE or "free").lower()
            if mode == "hybrid":
                summary = analyze_video_hybrid(base_path, transcript_text=None, max_frames=self.settings.FRAMES_MAX)
            else:
                summary = analyze_video_free_narrative(base_path, transcript_text=None, max_frames=self.settings.FRAMES_MAX)

            llm_calls = 1 if summary else 0
            if not summary:
                return EvaluateResponse(
                    duplicated=False, duplicate_reason=None, duplicate_candidate_url=None,
                    alignment=AlignmentResult(aproved=False, match_percent=0.0, reasons="No se pudo generar el resumen del video."),
                    cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                )

            # Texto compacto (útil si luego quieres persistir/embeber)
            _compact_text = summarize_video_textual(summary)

            # 3.b) Juez de alineación
            cmp_raw = comparar_descripcion_con_resumen_ia(req.descripcion, summary, umbral_aprobacion=70)
            if not cmp_raw:
                return EvaluateResponse(
                    duplicated=False, duplicate_reason=None, duplicate_candidate_url=None,
                    alignment=AlignmentResult(aproved=False, match_percent=0.0, reasons="No se pudo comparar la descripción con el resumen."),
                    cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
                )

            try:
                cmp_json = json.loads(cmp_raw)
            except Exception:
                cmp_json = {"aproved": False, "match_percent": 0.0, "reasons": "Respuesta IA inválida."}

            # 4) Registro de gasto
            guard.commit(req.campaign_id, spent_usd=0.02 if llm_calls else 0.0, used={"llm": llm_calls, "emb": 0})

            # 5) Persistir huellas del base si era MISS
            if not cached_base:
                try:
                    save_video_features(
                        campaign_id=req.campaign_id,
                        video_url=str(req.video_url),
                        phash64_bits=base_fp,
                        seq_bits=base_seq,
                        duration_s=get_duration_s(base_path),
                    )
                except Exception as e:
                    print(f"[warn] no se pudieron guardar huellas en Redis: {e}")

            # 6) Respuesta final
            return EvaluateResponse(
                duplicated=False,
                duplicate_reason=None,
                duplicate_candidate_url=None,
                alignment=AlignmentResult(
                    aproved=bool(cmp_json.get("aproved", False)),
                    match_percent=float(cmp_json.get("match_percent", 0.0)),
                    reasons=str(cmp_json.get("reasons", "Sin motivos."))
                ),
                cost={"llm_calls": llm_calls, "embedding_calls": 0, "transcription_seconds": 0, "degraded_path": False}
            )

        finally:
            # Limpieza de temporales (siempre)
            shutil.rmtree(root, ignore_errors=True)