from typing import Optional

def build_campaign_script_prompt(description: str, category: str, creator_type: str, requirements: Optional[str]) -> str:
    """
    Construye el prompt final para el LLM según el “contrato” acordado con backend Java.

    Reglas clave:
      - Duración objetivo: ~1 minuto.
      - Debe incluir HOOK viral (inicio) y CTA (cierre).
      - Ajuste según tipo de creador (UGC | CLIPPER/CLIPPING).
      - Permite instrucciones extra (no obligatorias).

    Parámetros:
      description: Brief/descripcion de la campaña.
      category: Categoría de la campaña (ej. GAMING, BEAUTY, FOOD, ...).
      creator_type: Tipo de creador (UGC | CLIPPER/CLIPPING | ...).
      requirements: Requisitos de la campaña

    Retorna:
      Cadena lista para ser enviada como mensaje del usuario al modelo.
    """
    # Normalizaciones defensivas (evitan espacios raros en el prompt)
    desc = (description or "").strip()
    cat = (category or "").strip()
    ctype = (creator_type or "").strip()
    req = (requirements or "").strip()

    type_clause = f"Tipo de creador: {ctype}."
    req_clause = f"\nRequisitos de campaña: {req}" if req else ""

    return (
        "Crea un guion preciso, de extensión media (~1 minuto) y claro para un video de campaña al que "
        "creadores UGC o Clipperos (según el tipo) van a postular. Debe tener:\n"
        "- Un HOOK VIRAL inicial (5-7s) y un CTA concreto al cierre.\n"
        "- Estructura en secciones (Hook, Desarrollo en 2-3 beats, CTA) con líneas y acciones visuales.\n"
        "- Lenguaje natural en español de LATAM, sin emojis, apto para TikTok/Reels/Shorts.\n"
        f"- Categoría: {cat}.\n"
        f"- Descripción/brief: {desc}.\n"
        f"- {type_clause}{req_clause}\n\n"
        "Devuelve SOLO el guion en formato Markdown con:\n"
        "### Hook\n"
        "### Desarrollo\n"
        "### CTA\n"
        "Incluye notas visuales entre [corchetes] cuando aporten claridad.\n"
        "NO envuelvas tu respuesta en triple backticks (```); responde solo en Markdown crudo."
    )