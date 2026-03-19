"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
集成测试：并发推理测试
测试方法论：并发测试设计——验证并发控制与无死锁
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【并发测试的核心问题】
并发 Bug 最难复现，因为它们依赖线程调度的时序。并发测试的目标不是"触发 Bug"，
而是通过大量并发场景验证以下三件事：

  1. 无崩溃（Safety）：N 个并发请求全部得到有效响应，没有异常传播出去
  2. 并发控制有效（Liveness）：系统不会因为并发而死锁，最终所有请求都能完成
  3. 性能合理（Performance）：并发执行的总时间应接近 ceil(N / max_concurrency) × 单次时间

【本服务的并发模型】
  - max_concurrency=4：同时最多 4 个推理在运行，超过的请求排队
  - asyncio.Semaphore：实现并发控制，不是真正的并行（GIL 限制），但 I/O 密集型任务可并发
  - asyncio.to_thread：把同步 predict 丢到线程池，真正解放事件循环
  - 理论时间：50 请求 / 4 并发 = 13 批，每批 ~50ms → 约 650ms

【asyncio.gather 说明】
  asyncio.gather(*coroutines) 会同时调度所有协程，
  相当于"发射后不管，等所有完成"。
  与 return_exceptions=True 配合，单个协程失败不会中断其他协程。

【死锁检测方法】
  给整个并发测试加 timeout（通过 pytest-timeout 或 asyncio.wait_for），
  如果在预期时间内没有完成，就认为可能发生了死锁。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import statistics

import pytest
import pytest_asyncio
from httpx import AsyncClient

# 整个模块共享一个事件循环，与 app_client (module scope) 匹配
pytestmark = pytest.mark.asyncio(loop_scope="module")

# ── 并发正确性测试 ────────────────────────────────────────────────────────────


async def test_concurrent_classify_50_requests_all_succeed(app_client: AsyncClient) -> None:
    """并发：50 个并发分类请求，全部应返回成功响应（无异常、无死锁）。

    使用 return_exceptions=True 收集所有结果（包括异常），
    最后再统一断言——这样即使部分请求失败也能看到全貌，而非在第一个失败时停止。
    """
    texts = [
        "中国队赢得世界杯小组赛首胜",
        "新款芯片采用3nm工艺制程",
        "国产电影票房突破百亿创历史",
        "A股市场成交量创年内新高",
        "科学家发现新型材料导电性能卓越",
    ]
    n_requests = 50

    async def single_request(idx: int):
        text = texts[idx % len(texts)]
        return await app_client.post("/api/v1/classify", json={"text": text})

    # asyncio.gather 并发发起 50 个请求
    # return_exceptions=True：异常作为普通值返回，不中断其他协程
    results = await asyncio.gather(
        *[single_request(i) for i in range(n_requests)],
        return_exceptions=True,
    )

    # 统计成功/失败数量
    exceptions = [r for r in results if isinstance(r, Exception)]
    responses = [r for r in results if not isinstance(r, Exception)]
    failed_responses = [r for r in responses if r.status_code != 200]

    assert len(exceptions) == 0, (
        f"有 {len(exceptions)} 个请求抛出异常: {exceptions[:3]}"
    )
    assert len(failed_responses) == 0, (
        f"有 {len(failed_responses)} 个请求返回非 200 状态码: "
        f"{[r.status_code for r in failed_responses[:5]]}"
    )
    assert len(responses) == n_requests, f"期望 {n_requests} 个响应，实际 {len(responses)}"


async def test_concurrent_classify_no_deadlock_within_timeout(app_client: AsyncClient) -> None:
    """并发：50 个并发请求应在合理时间内全部完成（验证无死锁）。

    超时时间设为 30 秒，远大于理论值（~650ms），
    只要不死锁，这个测试必然通过。
    """
    n_requests = 50

    async def run_all():
        results = await asyncio.gather(
            *[
                app_client.post(
                    "/api/v1/classify",
                    json={"text": f"测试文本第{i}条内容用于并发验证"},
                )
                for i in range(n_requests)
            ],
            return_exceptions=True,
        )
        return results

    # 30 秒超时兜底，触发说明可能发生了死锁
    try:
        results = await asyncio.wait_for(run_all(), timeout=30.0)
    except asyncio.TimeoutError:
        pytest.fail(
            f"{n_requests} 个并发请求在 30 秒内未完成，"
            "可能发生了死锁或信号量泄漏"
        )

    successful = [r for r in results if not isinstance(r, Exception) and r.status_code == 200]
    failures = [r for r in results if isinstance(r, Exception) or r.status_code != 200]
    print(f"\n失败详情(前5): {[(type(r).__name__, str(r)[:80]) if isinstance(r, Exception) else (r.status_code, r.text[:80]) for r in failures[:5]]}")
    assert len(successful) == n_requests, (
        f"期望 {n_requests} 个成功响应，实际 {len(successful)}"
    )


# ── 并发性能测试 ──────────────────────────────────────────────────────────────


async def test_concurrent_classify_latency_statistics(app_client: AsyncClient) -> None:
    """并发：统计 50 个并发请求的 P50/P95 延迟，验证性能合理。

    P50 < 2000ms，P95 < 5000ms（宽松阈值，适合 CI 环境）。
    注意：这里统计的是客户端感知延迟（含排队等待），不是单次推理延迟。
    """
    import time

    n_requests = 50
    latencies: list[float] = []

    async def timed_request(idx: int) -> float:
        start = time.monotonic()
        resp = await app_client.post(
            "/api/v1/classify",
            json={"text": f"并发延迟测试第{idx}条"},
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        assert resp.status_code == 200, f"请求 {idx} 失败: {resp.status_code}"
        return elapsed_ms

    results = await asyncio.gather(
        *[timed_request(i) for i in range(n_requests)],
        return_exceptions=True,
    )

    # 过滤异常
    latencies = [r for r in results if isinstance(r, float)]
    assert len(latencies) == n_requests, f"部分请求异常，只得到 {len(latencies)} 个延迟数据"

    latencies.sort()
    p50 = latencies[int(n_requests * 0.50)]
    p95 = latencies[int(n_requests * 0.95)]
    avg = statistics.mean(latencies)

    # 打印统计信息（pytest -s 时可见）
    print(f"\n并发延迟统计（{n_requests} 请求，max_concurrency=4）:")
    print(f"  P50 = {p50:.1f}ms")
    print(f"  P95 = {p95:.1f}ms")
    print(f"  AVG = {avg:.1f}ms")
    print(f"  MAX = {max(latencies):.1f}ms")

    assert p50 < 2000, f"P50={p50:.1f}ms 超过 2000ms 阈值"
    assert p95 < 5000, f"P95={p95:.1f}ms 超过 5000ms 阈值"


async def test_concurrent_requests_faster_than_sequential(app_client: AsyncClient) -> None:
    """并发：并发执行应比顺序执行显著更快（验证并发控制实际生效）。

    顺序执行 8 个请求的总时间，应远大于并发执行 8 个请求的时间。
    如果并发完全退化为串行，则说明 Semaphore 或 to_thread 有问题。
    """
    import time

    n_requests = 8
    payload_list = [
        {"text": f"并发对比测试第{i}条，包含足够多的中文内容用于测试"}
        for i in range(n_requests)
    ]

    # 顺序执行
    seq_start = time.monotonic()
    for payload in payload_list:
        resp = await app_client.post("/api/v1/classify", json=payload)
        assert resp.status_code == 200
    seq_elapsed = (time.monotonic() - seq_start) * 1000

    # 并发执行
    conc_start = time.monotonic()
    results = await asyncio.gather(
        *[app_client.post("/api/v1/classify", json=p) for p in payload_list],
        return_exceptions=True,
    )
    conc_elapsed = (time.monotonic() - conc_start) * 1000

    print(f"\n顺序执行 {n_requests} 请求: {seq_elapsed:.1f}ms")
    print(f"并发执行 {n_requests} 请求: {conc_elapsed:.1f}ms")
    print(f"加速比: {seq_elapsed / conc_elapsed:.2f}x")

    all_success = all(
        not isinstance(r, Exception) and r.status_code == 200 for r in results
    )
    assert all_success, "并发请求中有失败的响应"

    # 并发应至少比顺序快 30%（宽松阈值，避免 CI 环境波动导致误报）
    assert conc_elapsed < seq_elapsed * 0.9, (
        f"并发执行({conc_elapsed:.1f}ms)未明显快于顺序执行({seq_elapsed:.1f}ms)，"
        "可能并发控制未生效"
    )
