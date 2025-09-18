import time, os, json
from app.infrastructure.redisdb.client import redis

STREAM = "queue:inspect"
GROUP = "g1"
CONSUMER = os.getenv("HOSTNAME", "consumer-1")

def ensure_group():
    try:
        redis.xgroup_create(STREAM, GROUP, id="0-0", mkstream=True)
    except Exception:
        pass

def main():
    ensure_group()
    print("[worker] started")
    while True:
        resp = redis.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=10, block=5000)
        if not resp:
            continue
        for stream, messages in resp:
            for msg_id, fields in messages:
                try:
                    print(f"[worker] processing {msg_id} -> {fields}")
                    redis.xack(STREAM, GROUP, msg_id)
                except Exception as e:
                    print("[worker] error:", e)
        time.sleep(0.1)

if __name__ == "__main__":
    main()