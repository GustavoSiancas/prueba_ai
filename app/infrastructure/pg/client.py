import os
from typing import Optional
from psycopg_pool import ConnectionPool

_POOL: Optional[ConnectionPool] = None

def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is None:
        dsn = os.getenv("PG_DSN")
        if not dsn:
            raise RuntimeError("PG_DSN no definido (exporta PG_DSN o usa Settings.PG_DSN)")
        _POOL = ConnectionPool(conninfo=dsn, min_size=1, max_size=5, kwargs={"autocommit": True})
    return _POOL
