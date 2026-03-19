"""
单元测试：LRUTTLCache

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：状态转换测试（State Transition Testing）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
状态转换测试的核心思想：
  被测对象有明确的"状态"，测试用例覆盖所有状态及状态间的转换路径。

缓存条目的生命周期（状态机）：
  ┌────────────┐  put(k,v)  ┌──────────────┐
  │  不存在     │ ─────────► │   存在（活跃）  │
  └────────────┘            └──────┬───────┘
       ▲                           │ TTL 到期 / invalidate(k)
       │                           ▼
       │                    ┌──────────────┐
       │  LRU 淘汰           │   过期/删除   │
       └────────────────────┘
                    ▲
                    │ 超出 max_size
                    │（最旧的条目被淘汰）
                    │

需要覆盖的转换路径：
  ① 不存在 → 存在：put() 后 get() 命中
  ② 存在 → 不存在（TTL）：put() 后 sleep > TTL，get() 返回 None
  ③ 存在 → 不存在（LRU淘汰）：超过 max_size，最旧条目消失
  ④ 存在 → 不存在（invalidate）：主动删除
  ⑤ 访问刷新顺序：get() 应将条目移到"最近使用"末尾，改变 LRU 淘汰顺序

pytest 关键特性：
  fixture（cache_instance）— function scope，每测试独立，防止状态泄漏
  time.sleep() — 测试 TTL 过期（只睡 0.2s，代价小）
  threading.Thread — 验证线程安全性
"""

import threading
import time

import pytest

from src.services.cache_layer import LRUTTLCache


# ── 基本 get/put 测试 ─────────────────────────────────────────────────────────


def test_cache_put_and_get_returns_value(cache_instance: LRUTTLCache) -> None:
    """存入后能取出，且值一致。状态转换：不存在 → 存在。"""
    cache_instance.put("key1", "value1")
    result = cache_instance.get("key1")
    assert result == "value1", f"Expected 'value1', got {result!r}"


def test_cache_miss_returns_none(cache_instance: LRUTTLCache) -> None:
    """未命中返回 None（key 从未存入）。"""
    result = cache_instance.get("nonexistent_key")
    assert result is None, f"Expected None for missing key, got {result!r}"


def test_cache_put_overwrites_existing_value(cache_instance: LRUTTLCache) -> None:
    """重复 put 同一 key 应覆盖旧值。"""
    cache_instance.put("key1", "old_value")
    cache_instance.put("key1", "new_value")
    result = cache_instance.get("key1")
    assert result == "new_value", f"Expected 'new_value', got {result!r}"


def test_cache_supports_various_value_types(cache_instance: LRUTTLCache) -> None:
    """缓存应支持任意类型的值（dict、list、int）。"""
    cache_instance.put("dict_key", {"label": "sports", "confidence": 0.9})
    cache_instance.put("list_key", [1, 2, 3])
    cache_instance.put("int_key", 42)

    assert cache_instance.get("dict_key") == {"label": "sports", "confidence": 0.9}
    assert cache_instance.get("list_key") == [1, 2, 3]
    assert cache_instance.get("int_key") == 42


# ── LRU 淘汰测试 ──────────────────────────────────────────────────────────────


def test_cache_lru_evicts_oldest_when_full(cache_instance: LRUTTLCache) -> None:
    """容量为 3 时存入第 4 个，第 1 个（最久未用）应被淘汰。

    状态转换：存在 → 不存在（LRU 淘汰）
    cache_instance 的 max_size=3（见 conftest.py）
    """
    cache_instance.put("a", 1)
    cache_instance.put("b", 2)
    cache_instance.put("c", 3)
    # 存入第 4 个，触发 LRU 淘汰（"a" 最久未用）
    cache_instance.put("d", 4)

    assert cache_instance.get("a") is None, "'a' should have been evicted (LRU)"
    assert cache_instance.get("b") == 2, "'b' should still be in cache"
    assert cache_instance.get("c") == 3, "'c' should still be in cache"
    assert cache_instance.get("d") == 4, "'d' should be in cache"


def test_cache_lru_access_refreshes_order(cache_instance: LRUTTLCache) -> None:
    """get() 操作应将条目移到最近，从而改变 LRU 淘汰顺序。

    步骤：
    1. 存入 a, b, c（顺序：a最旧，c最新）
    2. 访问 a（现在顺序变为：b最旧，a最新）
    3. 存入 d（触发淘汰），b 应被淘汰而非 a
    """
    cache_instance.put("a", 1)
    cache_instance.put("b", 2)
    cache_instance.put("c", 3)

    # 访问 a，刷新其"最近使用"状态
    cache_instance.get("a")

    # 存入 d，b 应作为最久未用被淘汰
    cache_instance.put("d", 4)

    assert cache_instance.get("b") is None, "'b' should have been evicted after 'a' was accessed"
    assert cache_instance.get("a") == 1, "'a' should still be in cache (was recently accessed)"


def test_cache_eviction_count_tracked(cache_instance: LRUTTLCache) -> None:
    """LRU 淘汰次数应被正确统计在 stats 中。"""
    cache_instance.put("a", 1)
    cache_instance.put("b", 2)
    cache_instance.put("c", 3)
    cache_instance.put("d", 4)  # 触发 1 次淘汰

    stats = cache_instance.stats()
    assert stats["evictions"] == 1, (
        f"Expected 1 eviction, got {stats['evictions']}"
    )


# ── TTL 过期测试 ──────────────────────────────────────────────────────────────


def test_cache_ttl_expiry_returns_none() -> None:
    """TTL 过期后 get() 应返回 None。状态转换：存在 → 过期 → 不存在。

    使用独立的短 TTL 实例（不依赖 conftest 的 cache_instance）。
    """
    short_ttl_cache = LRUTTLCache(max_size=10, ttl_seconds=0.1)
    short_ttl_cache.put("key", "value")

    # 确认存在
    assert short_ttl_cache.get("key") == "value", "Key should exist before TTL expires"

    # 等待 TTL 过期（0.1s TTL + 0.1s buffer）
    time.sleep(0.2)

    result = short_ttl_cache.get("key")
    assert result is None, f"Expected None after TTL expiry, got {result!r}"


def test_cache_ttl_not_expired_still_accessible() -> None:
    """TTL 未到期时，值应仍可访问。"""
    cache = LRUTTLCache(max_size=10, ttl_seconds=5.0)
    cache.put("key", "value")

    # 立即访问（远未到期）
    result = cache.get("key")
    assert result == "value", "Value should be accessible before TTL expires"


# ── invalidate 测试 ───────────────────────────────────────────────────────────


def test_cache_invalidate_removes_key(cache_instance: LRUTTLCache) -> None:
    """invalidate() 后 get() 应返回 None。状态转换：存在 → 不存在（主动删除）。"""
    cache_instance.put("key", "value")
    result = cache_instance.invalidate("key")
    assert result is True, "invalidate should return True when key exists"
    assert cache_instance.get("key") is None, "Key should be gone after invalidate"


def test_cache_invalidate_nonexistent_returns_false(cache_instance: LRUTTLCache) -> None:
    """invalidate() 不存在的 key 应返回 False，不崩溃。"""
    result = cache_instance.invalidate("ghost_key")
    assert result is False, "invalidate should return False when key does not exist"


# ── stats 正确性测试 ──────────────────────────────────────────────────────────


def test_cache_stats_initial_state(cache_instance: LRUTTLCache) -> None:
    """初始状态：所有统计量应为 0，hit_rate 为 0.0。"""
    stats = cache_instance.stats()
    assert stats["total_requests"] == 0
    assert stats["hits"] == 0
    assert stats["evictions"] == 0
    assert stats["hit_rate"] == 0.0
    assert stats["current_size"] == 0


def test_cache_stats_hit_rate_calculation(cache_instance: LRUTTLCache) -> None:
    """hit_rate = hits / total_requests，应精确计算。

    步骤：put 一个 key，get 2次（1次命中，1次未命中）
    预期：hit_rate = 1/2 = 0.5
    """
    cache_instance.put("key", "value")
    cache_instance.get("key")         # 命中 (hit=1)
    cache_instance.get("missing_key") # 未命中 (hit=1, total=2)

    stats = cache_instance.stats()
    assert stats["total_requests"] == 2, f"Expected 2 requests, got {stats['total_requests']}"
    assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
    assert stats["hit_rate"] == pytest.approx(0.5), (
        f"Expected hit_rate=0.5, got {stats['hit_rate']}"
    )


def test_cache_stats_no_zero_division_with_no_requests(cache_instance: LRUTTLCache) -> None:
    """当 total_requests == 0 时，hit_rate 应为 0.0，不触发除零错误。"""
    stats = cache_instance.stats()
    assert stats["hit_rate"] == 0.0, "hit_rate should be 0.0 with no requests"


def test_cache_stats_current_size_updates(cache_instance: LRUTTLCache) -> None:
    """current_size 应随 put/invalidate 操作动态更新。"""
    assert cache_instance.stats()["current_size"] == 0

    cache_instance.put("a", 1)
    assert cache_instance.stats()["current_size"] == 1

    cache_instance.put("b", 2)
    assert cache_instance.stats()["current_size"] == 2

    cache_instance.invalidate("a")
    assert cache_instance.stats()["current_size"] == 1


# ── 线程安全测试 ──────────────────────────────────────────────────────────────


def test_cache_thread_safety_no_crash() -> None:
    """10 个线程同时读写不应崩溃或产生数据损坏。

    此测试验证 _lock 的正确性。
    不强断言最终状态（并发顺序不确定），只要不抛出异常即可。
    """
    shared_cache = LRUTTLCache(max_size=20, ttl_seconds=10.0)
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(10):
                key = f"key_{thread_id}_{i}"
                shared_cache.put(key, f"value_{i}")
                shared_cache.get(key)
                shared_cache.get("nonexistent")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety violation: {errors}"
