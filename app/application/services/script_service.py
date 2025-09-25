import os
import re
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI

from app.infrastructure.nlp.script_templates import build_campaign_script_prompt

FENCE_START = re.compile(r"^```(?:\w+)?\s*", re.IGNORECASE)
FENCE_END   = re.compile(r"\s*```$")

@dataclass
class ScriptGeneratorService:
    """
    Orquestador de la generación de guiones.

    Atributos:
     model: Nombre del modelo a usar en OpenAI (ej. "gpt-4o").
    """

    model: str = "gpt-4o"

    def _client(self) -> OpenAI:
        """
        Crea el cliente de OpenAI usando la API Key del entorno.

        Raises:
         RuntimeError: si OPENAI_API_KEY no está configurada.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no configurado")
        return OpenAI(api_key=api_key)

    def _strip_code_fences(self, content: str) -> str:
        """
        Quita fences de código (```... ```) si el modelo las devuelve.
        Deja solo el Markdown crudo para facilitar el render en Front.
        """
        if not content:
            return content
        content = FENCE_START.sub("", content).strip()
        content = FENCE_END.sub("", content).strip()
        return content

    def generate_script(self, description: str, category: str, creator_type: str, extra_prompt: Optional[str]) -> str:
        """
        Genera el guion final llamando al modelo de lenguaje.

        Parámetros:
          description, category, creator_type, extra_prompt: ver docstring del template.

        Retorna:
         El contenido del guion.

        Raises:
         RuntimeError: si la respuesta del modelo no contiene contenido utilizable.
        """
        # Pequeñas salvaguardas para evitar prompts vacíos o costos innecesarios.
        if not (description and category and creator_type):
            raise RuntimeError("Parámetros insuficientes para generar el guion")
        if len(description) > 2000:
            raise RuntimeError("Descripción demasiado larga (>2000 caracteres)")
        if extra_prompt and len(extra_prompt) > 1000:
            raise RuntimeError("Prompt extra demasiado largo (>1000 caracteres)")

        prompt = build_campaign_script_prompt(description, category, creator_type, extra_prompt)
        client = self._client()

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Eres un experto en marketing y publicidad."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        try:
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise RuntimeError(f"Respuesta inválida del modelo: {exc}")

        content = self._strip_code_fences(content)

        if not content:
            raise RuntimeError("El modelo no devolvió contenido")

        return content