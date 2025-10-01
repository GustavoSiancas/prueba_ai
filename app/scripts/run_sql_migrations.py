"""
Ejecutor de migraciones SQL en arranque.

- Lee todos los .sql de app/infrastructure/pg/migrations (orden alfabético).
- Se conecta con psycopg (v3) usando PG_DSN (añade sslmode=require si falta).
- Reintenta la conexión hasta que la DB esté disponible (útil en plataformas cloud).
- Idempotente si tus scripts lo son (CREATE IF NOT EXISTS, DROP IF EXISTS, etc.).
- Si PG_ENABLED=false, no hace nada y sale con status 0.
"""

import os
import sys
import time
import glob
from pathlib import Path
from typing import List

import psycopg


MIGRATIONS_DIR = os.getenv(
    "MIGRATIONS_DIR",
    "app/infrastructure/pg/migrations"
)


def _dsn_with_ssl(dsn: str) -> str:
    if "sslmode=" not in dsn:
        sep = "&" if "?" in dsn else "?"
        dsn = f"{dsn}{sep}sslmode=require"
    return dsn


def _wait_for_db_and_connect(dsn: str, attempts: int = 40, sleep_s: float = 2.0):
    last_err = None
    for i in range(attempts):
        try:
            conn = psycopg.connect(dsn, autocommit=False)
            return conn
        except Exception as e:
            last_err = e
            print(f"[migrations] intento {i+1}/{attempts} -> DB no lista aún: {e}", flush=True)
            time.sleep(sleep_s)
    print(f"[migrations] ERROR: no se pudo conectar a la DB luego de {attempts*sleep_s:.0f}s: {last_err}", flush=True)
    sys.exit(1)


def _list_sql_files(folder: str) -> List[Path]:
    base = Path(folder)
    if not base.exists():
        print(f"[migrations] carpeta no encontrada: {folder} (se omiten migraciones)", flush=True)
        return []
    files = sorted(Path(p) for p in glob.glob(str(base / "*.sql")))
    return files


def main():
    pg_enabled = os.getenv("PG_ENABLED", "false").lower() == "true"
    if not pg_enabled:
        print("[migrations] PG_ENABLED=false -> se omiten migraciones.", flush=True)
        return

    dsn = os.getenv("PG_DSN", "")
    if not dsn:
        print("[migrations] ERROR: PG_DSN vacío.", flush=True)
        sys.exit(1)

    dsn = _dsn_with_ssl(dsn)
    files = _list_sql_files(MIGRATIONS_DIR)
    if not files:
        print(f"[migrations] no hay archivos .sql en {MIGRATIONS_DIR}. Nada que aplicar.", flush=True)
        return

    print(f"[migrations] aplicando {len(files)} archivo(s) desde {MIGRATIONS_DIR}...", flush=True)

    # Espera a que la DB esté lista y conecta
    with _wait_for_db_and_connect(dsn) as conn:
        with conn.cursor() as cur:
            for fp in files:
                sql = fp.read_text(encoding="utf-8")
                if not sql.strip():
                    print(f"[migrations] {fp.name}: vacío. se omite.", flush=True)
                    continue
                try:
                    print(f"[migrations] -> {fp.name} ...", flush=True)
                    cur.execute(sql)  # psycopg3 soporta múltiples statements separados por ;
                    conn.commit()
                    print(f"[migrations] OK  {fp.name}", flush=True)
                except Exception as e:
                    conn.rollback()
                    print(f"[migrations] ERROR en {fp.name}: {e}", flush=True)
                    sys.exit(1)

    print("[migrations] todas las migraciones aplicadas correctamente.", flush=True)


if __name__ == "__main__":
    main()