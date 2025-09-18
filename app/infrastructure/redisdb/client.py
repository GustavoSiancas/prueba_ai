import os
import redis as _redis

"""
Cliente Redis compartido. Usa REDIS_URL o 'redis://localhost:6379/0'.
decode_responses=False porque guardamos binarios/JSON; decodificamos a mano.
"""

redis = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=False)