import os
from typing import Any

# app.config import 시 필수 환경변수 기본값 보장 (로컬 .env 부재 시에도 테스트 가능)
os.environ.setdefault("ENVIRONMENT", "dev")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173")
os.environ.setdefault("BACKEND_URL", "http://localhost:8080")
os.environ.setdefault("LLM_MODEL", "gemini-flash-lite-latest")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("EXA_API_KEY", "test-exa-key")


class FakePipeline:
    def __init__(self, redis: "FakeRedis"):
        self._redis = redis
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def rpush(self, key: str, *values: str):
        self._ops.append(("rpush", (key, *values), {}))
        return self

    def ltrim(self, key: str, start: int, end: int):
        self._ops.append(("ltrim", (key, start, end), {}))
        return self

    def expire(self, key: str, ttl: int):
        self._ops.append(("expire", (key, ttl), {}))
        return self

    def delete(self, key: str):
        self._ops.append(("delete", (key,), {}))
        return self

    def zadd(self, key: str, mapping: dict[str, float]):
        self._ops.append(("zadd", (key, mapping), {}))
        return self

    def zremrangebyrank(self, key: str, min_rank: int, max_rank: int):
        self._ops.append(("zremrangebyrank", (key, min_rank, max_rank), {}))
        return self

    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False):
        self._ops.append(("set", (key, value), {"ex": ex, "nx": nx}))
        return self

    async def execute(self) -> list[Any]:
        results = []
        for name, args, kwargs in self._ops:
            fn = getattr(self._redis, name)
            results.append(await fn(*args, **kwargs))
        self._ops.clear()
        return results


class FakeRedis:
    """단위 테스트용 인메모리 비동기 Redis 스텁"""

    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.ttls: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def incr(self, key: str) -> int:
        val = int(self.kv.get(key, "0")) + 1
        self.kv[key] = str(val)
        return val

    async def sadd(self, key: str, *values: str) -> int:
        bucket = self.sets.setdefault(key, set())
        before = len(bucket)
        bucket.update(values)
        return len(bucket) - before

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    async def srem(self, key: str, *values: str) -> int:
        bucket = self.sets.get(key, set())
        removed = 0
        for v in values:
            if v in bucket:
                bucket.remove(v)
                removed += 1
        return removed

    async def zincrby(self, key: str, amount: float, value: str) -> float:
        bucket = self.zsets.setdefault(key, {})
        bucket[value] = bucket.get(value, 0.0) + float(amount)
        return bucket[value]

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.lists.get(key, [])
        if not items:
            return []
        if end == -1:
            return list(items[start:])
        return list(items[start : end + 1])

    async def rpush(self, key: str, *values: str) -> int:
        bucket = self.lists.setdefault(key, [])
        bucket.extend(values)
        return len(bucket)

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        items = self.lists.get(key, [])
        if not items:
            return True
        if end == -1:
            self.lists[key] = items[start:]
        else:
            self.lists[key] = items[start : end + 1]
        return True

    async def expire(self, key: str, ttl: int) -> bool:
        self.ttls[key] = ttl
        return True

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def exists(self, key: str) -> int:
        return int(
            key in self.kv or key in self.lists or key in self.zsets or key in self.sets
        )

    async def delete(self, key: str) -> int:
        removed = 0
        for store in (self.kv, self.lists, self.zsets, self.sets):
            if key in store:
                del store[key]
                removed = 1
        return removed

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        bucket = self.zsets.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False,
    ) -> list[Any]:
        bucket = self.zsets.get(key, {})
        ordered = sorted(bucket.items(), key=lambda kv: kv[1])
        if not ordered:
            return []
        sliced = ordered[start:] if end == -1 else ordered[start : end + 1]
        if withscores:
            return sliced
        return [m for m, _ in sliced]

    async def zremrangebyrank(self, key: str, min_rank: int, max_rank: int) -> int:
        bucket = self.zsets.get(key, {})
        ordered = sorted(bucket.items(), key=lambda kv: kv[1])
        n = len(ordered)
        if n == 0:
            return 0
        s = min_rank if min_rank >= 0 else max(0, n + min_rank)
        e = max_rank if max_rank >= 0 else n + max_rank
        if e < s or s >= n:
            return 0
        to_remove = ordered[s : e + 1]
        for member, _ in to_remove:
            bucket.pop(member, None)
        return len(to_remove)
