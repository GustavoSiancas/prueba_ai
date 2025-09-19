from datetime import datetime
from typing import Literal
from .client import redis

"""
Guardarraíles de presupuesto/cotas por campaña/día (en Redis).
- snapshot(): lectura de contadores.
- decide(): 'allow'/'degrade' según proyección de consumo.
- commit(): incrementa contadores con TTL ~36h.
"""

BudgetAction = Literal["allow", "degrade", "deny"]

class BudgetGuardrails:
    """
    Reglas de presupuesto:
      - Cap de USD/día
      - Cap de llamadas LLM/día
      - Cap de llamadas Embeddings/día
    """

    def __init__(self, usd_day_cap: float, max_llm_calls: int, max_emb_calls: int):
        self.usd_day_cap = usd_day_cap
        self.max_llm_calls = max_llm_calls
        self.max_emb_calls = max_emb_calls

    def _keys(self, campaign_id: str):
        day = datetime.utcnow().strftime("%Y%m%d")
        base = f"cost:{campaign_id}:day:{day}"
        return base

    def snapshot(self, campaign_id: str) -> dict:
        """Lee los contadores actuales (usd, llm, emb)."""

        data = redis.hgetall(self._keys(campaign_id))
        snap = {}
        for k, v in data.items():
            k = k.decode() if isinstance(k, bytes) else k
            try:
                snap[k] = float(v.decode() if isinstance(v, bytes) else v)
            except:
                snap[k] = 0.0
        return snap

    def decide(self, campaign_id: str, est_usd: float, need: dict) -> BudgetAction:
        """
        Evalúa si se permite un gasto estimado y consumo de recursos.

        Returns:
          "allow" | "degrade" (evitar IA cara) | "deny" (reservado).
        """

        k = self._keys(campaign_id)
        snap = self.snapshot(campaign_id)
        usd_used = snap.get("usd", 0.0)
        llm = snap.get("llm", 0.0)
        emb = snap.get("emb", 0.0)

        if usd_used + est_usd > self.usd_day_cap:
            return "degrade"
        if llm + need.get("llm", 0) > self.max_llm_calls:
            return "degrade"
        if emb + need.get("emb", 0) > self.max_emb_calls:
            return "degrade"
        return "allow"

    def commit(self, campaign_id: str, spent_usd: float, used: dict):
        """Actualiza contadores (con TTL ~36h)."""

        k = self._keys(campaign_id)
        pipe = redis.pipeline()
        pipe.hincrbyfloat(k, "usd", spent_usd)
        if used.get("llm"): pipe.hincrbyfloat(k, "llm", used["llm"])
        if used.get("emb"): pipe.hincrbyfloat(k, "emb", used["emb"])
        pipe.expire(k, 60*60*36)
        pipe.execute()
