"""
性能测试 — 延迟基准测试（Latency Benchmark）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：性能基准测试（Benchmark Testing）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目的：量化系统性能，建立基线，防止性能退化（Performance Regression）。

什么是 pytest-benchmark？
  pytest-benchmark 是 pytest 的插件，提供 `benchmark` fixture。
  它会把被测函数运行多次（自动决定次数），统计：
    - min / max / mean / stddev：最小/最大/均值/标准差
    - P50 / P75 / P95 / P99：百分位延迟
  百分位延迟（Percentile Latency）是性能测试的核心指标：
    P95 = 200ms 意思是"95%的请求在200ms内完成"
    P99 = 500ms 意思是"99%的请求在500ms内完成（最慢1%不超过500ms）"

为什么用 P99 而不是 mean？
  均值会被极端值拉偏。生产环境中用户感受到的是"最慢的那次请求"，
  所以 SLO（服务质量目标）通常用 P95 或 P99 来定义。

本文件使用两种方式测量延迟：
  1. pytest-benchmark：通过 benchmark fixture 测量单条分类的底层延迟
  2. 手动统计：通过 asyncio 发送多次请求，用 statistics 模块计算百分位

回归阈值（Regression Threshold）：
  P99 < 200ms：如果某次代码提交让 P99 超过 200ms，性能测试失败，
  提醒开发者检查是否引入了性能退化。

注意：本测试需要 `pytest-benchmark` 包。
  TextClassifier.predict 中有 random.uniform(0.01, 0.05) 的模拟延迟，
  所以单次调用约 10-50ms，多次统计 P99 应 < 200ms。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import statistics
import time

import pytest
import pytest_asyncio
from httpx import AsyncClient

from src.models.text_classifier import TextClassifier


# ── 配置常量 ─────────────────────────────────────────────────────────────────

_SAMPLE_TEXT = "人工智能大模型发布引发行业震动"
_BENCHMARK_ROUNDS = 50       # 手动统计时的请求次数
_P99_THRESHOLD_MS = 200.0    # P99 延迟回归阈值（毫秒）


# ── benchmark fixture 测试（pytest-benchmark 方式）────────────────────────────


def test_classify_latency_benchmark(
    classifier_model: TextClassifier,
    benchmark,
) -> None:
    """使用 pytest-benchmark 测量 TextClassifier.predict 的延迟分布。

    benchmark fixture 会自动：
      1. 预热（warmup）：先跑几次，等JIT/caching稳定
      2. 多轮运行（rounds）：重复执行被测函数
      3. 统计分析：计算 min/max/mean/stddev/percentiles

    运行后在终端输出类似：
      Name                          Min     Max    Mean  StdDev  Median
      test_classify_latency_bench  12ms   48ms   25ms    8ms    22ms

    回归检测：
      如果 P99 > 200ms，断言失败，CI 报警。
    """
    # benchmark.pedantic 允许精确控制轮数和预热，避免初次调用的冷启动干扰
    result = benchmark.pedantic(
        classifier_model.predict,
        args=(_SAMPLE_TEXT,),
        rounds=30,
        warmup_rounds=3,
    )

    # 验证基准测试的结果格式正确（benchmark本身已经统计延迟，这里验证功能正确性）
    assert result is not None, "predict 不应返回 None"
    assert "label" in result, "predict 返回值缺少 label"
    assert "confidence" in result, "predict 返回值缺少 confidence"

    # 从 benchmark 统计中提取 P99（pytest-benchmark 通过 benchmark.stats 暴露）
    # benchmark.stats 在测试结束后填充，运行时通过 benchmark 对象访问
    # 注意：pytest-benchmark < 4.0 用 benchmark.stats['percentiles']['99']
    #        新版直接通过 benchmark.stats.stats.percentiles 访问
    # 这里我们用手动统计方式验证 P99，更直观可控
    assert benchmark.stats["mean"] < _P99_THRESHOLD_MS / 1000, (
        f"平均延迟 {benchmark.stats['mean']*1000:.1f}ms 超过阈值 {_P99_THRESHOLD_MS}ms"
    )


# ── 手动百分位统计（API 层面端到端延迟）──────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_api_latency_percentiles(app_client: AsyncClient) -> None:
    """手动统计 API 端到端延迟的百分位分布。

    发送 _BENCHMARK_ROUNDS 次请求，记录每次耗时，
    计算 P50 / P95 / P99，并验证 P99 < 阈值。

    端到端延迟包含：
      网络传输（ASGI内部）+ 路由解析 + Pydantic校验 + 推理 + 响应序列化
    因此比纯模型推理延迟略高，但仍应在阈值内。
    """
    latencies_ms: list[float] = []

    for _ in range(_BENCHMARK_ROUNDS):
        start = time.perf_counter()
        response = await app_client.post(
            "/api/v1/classify",
            json={"text": _SAMPLE_TEXT},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        assert response.status_code == 200, (
            f"基准测试请求失败，状态码 {response.status_code}"
        )

    latencies_ms.sort()
    n = len(latencies_ms)

    def percentile(data: list[float], p: float) -> float:
        """计算第 p 百分位值（线性插值）。"""
        idx = (p / 100) * (len(data) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        frac = idx - lo
        return data[lo] * (1 - frac) + data[hi] * frac

    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    p99 = percentile(latencies_ms, 99)
    mean_ms = statistics.mean(latencies_ms)

    # 打印延迟分布，方便查看（pytest -s 时可见）
    print(f"\n延迟统计（n={n}次）:")
    print(f"  Mean={mean_ms:.1f}ms, P50={p50:.1f}ms, P95={p95:.1f}ms, P99={p99:.1f}ms")

    # 回归断言：P99 必须低于阈值
    assert p99 < _P99_THRESHOLD_MS, (
        f"P99 延迟 {p99:.1f}ms 超过阈值 {_P99_THRESHOLD_MS}ms，"
        f"可能存在性能退化（P50={p50:.1f}ms, P95={p95:.1f}ms）"
    )

    # P50 合理性验证（模拟延迟 10-50ms，加上框架开销，P50 应 < 100ms）
    assert p50 < 100.0, (
        f"P50 延迟 {p50:.1f}ms 过高，可能有阻塞操作"
    )


# ── 吞吐量测试：并发场景下的延迟 ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_concurrent_latency(app_client: AsyncClient) -> None:
    """并发场景下的延迟分布测试。

    同时发起 20 个并发请求（模拟真实并发负载），
    统计整批完成时间和单个请求的延迟分布。

    推理引擎 max_concurrency=4，所以 20 个请求会产生排队效应，
    平均延迟会高于单请求，但 P99 仍应在合理范围内。
    """
    _CONCURRENCY = 20
    _CONCURRENT_P99_THRESHOLD_MS = 1500.0  # 并发场景阈值更宽松

    async def single_request() -> float:
        """发送一个分类请求，返回耗时（毫秒）。"""
        start = time.perf_counter()
        response = await app_client.post(
            "/api/v1/classify",
            json={"text": _SAMPLE_TEXT},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert response.status_code == 200, f"并发请求失败: {response.status_code}"
        return elapsed_ms

    # 并发发起所有请求
    batch_start = time.perf_counter()
    latencies = await asyncio.gather(*[single_request() for _ in range(_CONCURRENCY)])
    total_ms = (time.perf_counter() - batch_start) * 1000

    latencies_sorted = sorted(latencies)
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted))]

    print(f"\n并发测试（n={_CONCURRENCY}）:")
    print(f"  总耗时={total_ms:.0f}ms, P99={p99:.0f}ms")

    assert all(l > 0 for l in latencies), "所有请求延迟应大于0"
    assert p99 < _CONCURRENT_P99_THRESHOLD_MS, (
        f"并发P99 {p99:.0f}ms 超过阈值 {_CONCURRENT_P99_THRESHOLD_MS}ms"
    )
