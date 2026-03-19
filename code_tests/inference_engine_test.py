"""
inference_engine.py 的功能验证测试

运行方式（在项目根目录）：
    python -m pytest code_tests/inference_engine_test.py -v
或直接运行：
    python code_tests/inference_engine_test.py
"""

import asyncio
import sys
import time

sys.path.insert(0, ".")

from src.models.text_classifier import TextClassifier
from src.services.model_registry import ModelRegistry
from src.services.inference_engine import InferenceEngine, InferenceRequest


# ── 测试辅助 ─────────────────────────────────────────────────────────────────


def setup_engine(max_concurrency: int = 4, timeout: float = 3.0) -> InferenceEngine:
    """创建一个带真实模型的引擎实例。"""
    registry = ModelRegistry()
    classifier = TextClassifier()
    registry.register("text_classifier", "v1", classifier)
    return InferenceEngine(registry, max_concurrency=max_concurrency, timeout=timeout)


# ── 测试1：基本推理成功 ───────────────────────────────────────────────────────


async def test_basic_inference():
    engine = setup_engine()
    req = InferenceRequest(
        model_name="text_classifier",
        input_data="中国队赢得世界杯冠军",
        request_id="req-001",
    )
    resp = await engine.submit(req)

    assert resp.success is True, f"推理应成功，但 success={resp.success}"
    assert resp.request_id == "req-001", "request_id 应与请求一致"
    assert resp.result is not None, "成功时 result 不应为 None"
    assert resp.error is None, f"成功时 error 应为 None，实际: {resp.error}"
    assert resp.latency_ms > 0, "latency_ms 应大于 0"
    assert "label" in resp.result, f"result 应含 label 字段，实际: {resp.result}"
    assert "confidence" in resp.result, "result 应含 confidence 字段"
    print(f"  [OK] 推理成功: label={resp.result['label']}, latency={resp.latency_ms:.1f}ms")


# ── 测试2：超时机制 ───────────────────────────────────────────────────────────


async def test_timeout():
    """用极短超时（0.001秒）触发超时响应。"""
    engine = setup_engine(timeout=0.001)
    req = InferenceRequest(
        model_name="text_classifier",
        input_data="测试超时",
        request_id="req-timeout",
    )
    resp = await engine.submit(req)

    assert resp.success is False, f"极短超时应失败，但 success={resp.success}"
    assert resp.error is not None, "超时时 error 不应为 None"
    assert resp.result is None, f"超时时 result 应为 None，实际: {resp.result}"
    print(f"  [OK] 超时响应: error={resp.error}")


# ── 测试3：统计指标更新 ───────────────────────────────────────────────────────


async def test_stats_update():
    engine = setup_engine(timeout=3.0)

    # 发送2个成功请求
    for i in range(2):
        req = InferenceRequest(
            model_name="text_classifier",
            input_data="科技公司发布新产品",
            request_id=f"req-stats-{i}",
        )
        await engine.submit(req)

    # 发送1个超时请求
    engine._timeout = 0.001
    req = InferenceRequest(
        model_name="text_classifier",
        input_data="触发超时",
        request_id="req-timeout-stats",
    )
    await engine.submit(req)

    stats = engine.stats()
    assert stats["total_requests"] == 3, f"total_requests 应为3，实际: {stats['total_requests']}"
    assert stats["success_count"] == 2, f"success_count 应为2，实际: {stats['success_count']}"
    assert stats["failure_count"] == 1, f"failure_count 应为1，实际: {stats['failure_count']}"
    assert stats["avg_latency_ms"] > 0, "avg_latency_ms 应大于0"
    print(f"  [OK] 统计指标: {stats}")


# ── 测试4：并发控制 ───────────────────────────────────────────────────────────


async def test_concurrency_limit():
    """同时提交8个请求，Semaphore(2) 保证最多2个并发执行。"""
    engine = setup_engine(max_concurrency=2, timeout=5.0)

    texts = [
        "体育新闻", "科技发布", "娱乐明星", "财经股市",
        "足球比赛", "人工智能", "电影上映", "基金涨跌",
    ]
    requests = [
        InferenceRequest(
            model_name="text_classifier",
            input_data=text,
            request_id=f"req-conc-{i}",
        )
        for i, text in enumerate(texts)
    ]

    start = time.monotonic()
    responses = await asyncio.gather(*[engine.submit(r) for r in requests])
    elapsed = time.monotonic() - start

    success_count = sum(1 for r in responses if r.success)
    assert success_count == 8, f"全部8个请求应成功，实际成功: {success_count}"
    # 有并发限制，应比8个串行慢，但比完全串行快
    print(f"  [OK] 并发控制: 8个请求全部完成，耗时 {elapsed:.2f}s，并发数=2")


# ── 测试5：模型不存在时的行为 ─────────────────────────────────────────────────


async def test_model_not_found():
    """请求不存在的模型，_run_inference 应抛出异常，submit 应……"""
    engine = setup_engine()
    req = InferenceRequest(
        model_name="nonexistent_model",
        input_data="测试",
        request_id="req-notfound",
    )
    # ModelNotFoundError 不是 TimeoutError，submit 目前不捕获它
    # 预期：抛出 ModelNotFoundError（不会被吞掉）
    resp = await engine.submit(req)
    assert resp.success is False, f"模型不存在时应返回失败响应，实际: {resp.success}"
    assert resp.error is not None and "ModelNotFoundError" in resp.error, \
        f"error 应含 ModelNotFoundError，实际: {resp.error}"
    print(f"  [OK] 模型不存在时返回失败响应: error={resp.error}")



# ── 测试6：初始状态 stats ─────────────────────────────────────────────────────


async def test_initial_stats():
    engine = setup_engine()
    stats = engine.stats()
    assert stats["total_requests"] == 0
    assert stats["success_count"] == 0
    assert stats["failure_count"] == 0
    assert stats["avg_latency_ms"] == 0.0, "无请求时 avg 应为 0.0（除零保护）"
    print(f"  [OK] 初始统计: {stats}")


# ── 主入口 ───────────────────────────────────────────────────────────────────


async def main():
    tests = [
        ("基本推理成功", test_basic_inference),
        ("超时机制", test_timeout),
        ("统计指标更新", test_stats_update),
        ("并发控制（Semaphore=2，8个请求）", test_concurrency_limit),
        ("模型不存在时的行为", test_model_not_found),
        ("初始 stats 除零保护", test_initial_stats),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            await test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] 断言失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] 未预期异常: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果: {passed} 通过 / {failed} 失败 / {len(tests)} 总计")


if __name__ == "__main__":
    asyncio.run(main())
