"""
单元测试：TokenBucketRateLimiter

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：时序相关测试 + Mock 时间
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
时序相关测试的挑战：
  令牌桶依赖真实时间（time.monotonic()）来计算补充令牌数。
  直接测试会产生两个问题：
    1. 测试需要真实等待（sleep），导致测试变慢
    2. 测试结果受 CPU 调度影响，不稳定（flaky）

解决方案：Mock 时间
  用 unittest.mock.patch('time.monotonic') 替换真实时钟，
  让测试能"瞬间"模拟时间流逝。

  原理类比：想测一台每小时补充10颗糖的糖果机，
           不需要真等1小时，直接把机器内部时钟拨快1小时即可。

  注意：patch 的路径必须是"被测模块实际调用的路径"：
    ✓  'src.middleware.rate_limiter.time.monotonic'
    ✗  'time.monotonic'（只替换 time 模块，不影响已 import 的引用）

pytest 关键特性：
  unittest.mock.patch — 上下文管理器，替换模块中的函数/属性
  MagicMock.side_effect — 让 mock 每次调用返回不同的值（模拟时间递增）
"""

from unittest.mock import patch

import pytest

from src.middleware.rate_limiter import TokenBucketRateLimiter


# ── 基本令牌消耗测试 ──────────────────────────────────────────────────────────


def test_rate_limiter_allows_up_to_capacity(rate_limiter: TokenBucketRateLimiter) -> None:
    """容量为 5 时，前 5 次 allow() 应全部返回 True。

    conftest 中 rate_limiter 的 capacity=5。
    """
    results = [rate_limiter.allow("client_a") for _ in range(5)]
    assert all(results), (
        f"Expected all True for first 5 requests, got: {results}"
    )


def test_rate_limiter_blocks_after_capacity_exhausted(
    rate_limiter: TokenBucketRateLimiter,
) -> None:
    """令牌耗尽后，第 6 次应返回 False。"""
    for _ in range(5):
        rate_limiter.allow("client_a")  # 耗尽令牌

    result = rate_limiter.allow("client_a")
    assert result is False, "Should be blocked when tokens are exhausted"


def test_rate_limiter_exact_boundary(rate_limiter: TokenBucketRateLimiter) -> None:
    """精确边界：第 5 次允许，第 6 次拒绝。"""
    for i in range(4):
        assert rate_limiter.allow("client_x") is True, f"Request {i+1} should be allowed"

    assert rate_limiter.allow("client_x") is True, "Request 5 (at capacity) should be allowed"
    assert rate_limiter.allow("client_x") is False, "Request 6 (over capacity) should be blocked"


# ── Mock 时间测试：令牌恢复 ───────────────────────────────────────────────────


def test_rate_limiter_refills_after_time_passes() -> None:
    """时间推进后，令牌应恢复，再次允许请求。

    Mock 策略：
      第1次调用 time.monotonic() → 返回 0.0（初始化桶）
      第2次调用 time.monotonic() → 返回 0.0（第1个请求）
      ...（耗尽令牌）
      之后调用 time.monotonic() → 返回 10.0（模拟10秒后）

    capacity=5, refill_rate=0.5/s：
      10秒补充 = 10 * 0.5 = 5 个令牌，桶再次满
    """
    limiter = TokenBucketRateLimiter(capacity=5, refill_rate=0.5)

    # 用 side_effect 列表：前 N 次调用monotonic会改为依次返回列表中的值
    # 初始化桶需要一次 time.monotonic()，每次 allow() 需要一次
    # 总共：1（初始化）+ 5（耗尽）= 6 次 → 都返回 0.0
    # 第 7 次（下一次 allow）→ 返回 10.0（时间推进10秒）
    time_values = [0.0] * 6 + [10.0] * 10  # 足够多的值

    with patch("src.middleware.rate_limiter.time.monotonic", side_effect=time_values):
        # 耗尽令牌
        for _ in range(5):
            limiter.allow("client")

        # 第 6 次被拒绝（此时时间还是 0.0）
        # 注：由于 side_effect 列表，这次会消耗一个 0.0，此时总共消耗了 7 个 0.0（初始化 + 5 次 allow + 1 次 allow）
        assert limiter.allow("client") is False, "Should be blocked before time passes"

        # 时间推进到 10.0 秒（使用列表中的 10.0 值）
        # 10s * 0.5/s = 5 tokens 补充，允许通过
        result = limiter.allow("client")
        assert result is True, "Should be allowed after tokens refilled"


def test_rate_limiter_partial_refill() -> None:
    """部分时间流逝应只恢复相应数量的令牌。

    capacity=2, refill_rate=1.0/s：
      耗尽 2 个令牌后，等 1 秒应恢复 1 个令牌（不是全部）
    """
    limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1.0)

    time_values = [0.0] * 3 + [1.0] * 10  # 先 0.0，后 1.0

    with patch("src.middleware.rate_limiter.time.monotonic", side_effect=time_values):
        # 耗尽 2 个令牌，monotonic调用 3 次（1次初始化 + 2次 allow）
        limiter.allow("client")
        limiter.allow("client")

        # 立即请求被拒绝（仍在 0.0）
        assert limiter.allow("client") is False, "Should be blocked"

        # 1秒后：恢复 1 个令牌，刚好能通过 1 次
        assert limiter.allow("client") is True, "Should be allowed with 1 refilled token"

        # 第 2 次还是被拒绝（只恢复了 1 个）
        assert limiter.allow("client") is False, "Should be blocked again (only 1 token refilled)"


# ── 多客户端独立测试 ──────────────────────────────────────────────────────────


def test_rate_limiter_different_clients_independent(
    rate_limiter: TokenBucketRateLimiter,
) -> None:
    """不同 client_id 的令牌桶互相独立。

    client_a 耗尽令牌，不影响 client_b 的配额。
    """
    # 耗尽 client_a 的令牌
    for _ in range(5):
        rate_limiter.allow("client_a")

    # client_a 被拒绝
    assert rate_limiter.allow("client_a") is False, "client_a should be blocked"

    # client_b 不受影响，仍有满桶令牌
    assert rate_limiter.allow("client_b") is True, (
        "client_b should be unaffected by client_a's exhaustion"
    )


def test_rate_limiter_multiple_clients_each_limited(
    rate_limiter: TokenBucketRateLimiter,
) -> None:
    """多个客户端各自有独立限流，都能被正确限速。"""
    clients = ["client_1", "client_2", "client_3"]

    for client in clients:
        # 每个客户端前 5 次通过
        for _ in range(5):
            assert rate_limiter.allow(client) is True, f"{client} should be allowed"
        # 第 6 次被拒绝
        assert rate_limiter.allow(client) is False, f"{client} should be blocked"


# ── reset 测试 ────────────────────────────────────────────────────────────────


def test_rate_limiter_reset_restores_full_capacity(
    rate_limiter: TokenBucketRateLimiter,
) -> None:
    """reset() 后，客户端令牌恢复为满桶。"""
    # 耗尽令牌
    for _ in range(5):
        rate_limiter.allow("client")
    assert rate_limiter.allow("client") is False, "Should be blocked before reset"

    # 重置
    rate_limiter.reset("client")

    # 重置后应能通过
    assert rate_limiter.allow("client") is True, "Should be allowed after reset"


# ── get_bucket_state 测试 ─────────────────────────────────────────────────────


def test_rate_limiter_bucket_state_decreases_after_request(
    rate_limiter: TokenBucketRateLimiter,
) -> None:
    """每次 allow() 后，桶中令牌数应减少 1。"""
    initial_state = rate_limiter.get_bucket_state("client")
    initial_tokens = initial_state["tokens"]  # 5.0（满桶）

    rate_limiter.allow("client")
    state_after = rate_limiter.get_bucket_state("client")

    assert state_after["tokens"] < initial_tokens, (
        f"Tokens should decrease after allow(). "
        f"Before: {initial_tokens}, After: {state_after['tokens']}"
    )


def test_rate_limiter_new_client_has_full_bucket(
    rate_limiter: TokenBucketRateLimiter,
) -> None:
    """新客户端（从未请求过）应有满桶令牌。"""
    state = rate_limiter.get_bucket_state("brand_new_client")
    assert state["tokens"] == pytest.approx(5.0), (
        f"New client should have full bucket (5.0), got {state['tokens']}"
    )
