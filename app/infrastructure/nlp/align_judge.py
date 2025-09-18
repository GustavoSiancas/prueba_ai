import os, json
from openai import OpenAI

"""
Juez de alineación:
- Compara la narrativa/JSON del resumen VLM con la descripción de campaña.
- Calcula match_percent y aprueba si supera el umbral.
"""

def comparar_descripcion_con_resumen_ia(descripcion: str, resumen, umbral_aprobacion: int = 70):
    """
    Ejecuta un prompt estricto que:
      - Extrae requisitos atómicos del brief.
      - Marca met/partial/not_met usando SOLO la evidencia del resumen.
      - Calcula match_percent y aproved.

    Args:
        descripcion: texto de la campaña.
        resumen: str (narrative) o dict (hybrid JSON).
        umbral_aprobacion: % para aproved=True.

    Returns:
        JSON string con {match_percent, aproved, reasons}.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if isinstance(resumen, dict):
        resumen_str = json.dumps(resumen, ensure_ascii=False)
    else:
        resumen_str = str(resumen)

    prompt = f"""
    Eres un evaluador ESTRICTO de cumplimiento de campaña.
    Compara el video SOLO con lo que la campaña pide. No penalices extras.
    Usa el texto/narrative y/o JSON provisto como ÚNICA evidencia. Si algo no consta, trátalo como desconocido.

    PASOS:
    1) Extrae requisitos atómicos.
    2) Para cada uno: met/partial/not_met según evidencia explícita.
    3) match_percent = round((met + 0.5*partial)/max(total,1)*100,2); aproved = match_percent >= {umbral_aprobacion}.

    ENTRADAS
    Descripción de campaña:
    \"\"\"{descripcion}\"\"\"

    Resumen del video (texto o JSON):
    \"\"\"{resumen_str}\"\"\"

    SALIDA JSON:
    {{
        "match_percent": 0-100,
        "aproved": true/false,
        "reasons": "2–3 oraciones; español; sin listas."
    }}
    """.strip()

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content": prompt}],
        response_format={"type":"json_object"},
        temperature=0.1
    )
    return resp.choices[0].message.content