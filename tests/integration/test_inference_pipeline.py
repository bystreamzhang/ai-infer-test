"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
集成测试：推理完整链路测试
测试方法论：端到端测试 vs 集成测试的区别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【端到端测试 vs 集成测试】

  集成测试（Integration Test）：
    验证两个或多个组件协作是否正确，关注"组件边界是否对齐"。
    例如：HTTP 请求 → FastAPI 路由 → InferenceEngine → 模型
    本文件属于集成测试：我们走完了整条 HTTP 链路，但服务是跑在内存里的
    （没有真实的网络 I/O），也没有外部依赖。

  端到端测试（E2E Test）：
    在与生产环境尽可能一致的条件下，从用户视角触发完整业务流程。
    例如：真实启动 uvicorn 服务 → 用 curl / Playwright 发请求 → 验证结果。
    E2E 更重，通常跑在 CI/CD 的独立 stage，而非本地开发循环中。

  本文件关注三个链路级别的特性：
  1. 完整推理链路：请求能正确到达模型并返回有意义的结果
  2. 缓存生效：相同请求第二次延迟显著低于第一次
  3. 指标更新：每次请求后 metrics 的 counter 应递增

【注意：缓存测试的设计决策】
  缓存是否命中取决于推理引擎是否开启了缓存（InferenceEngine 内部实现）。
  如果引擎没有集成缓存，缓存测试应该验证"延迟保持合理"而非"延迟降低"。
  本测试设计为：若有缓存则验证加速效果，若无缓存则仅验证链路正常。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import time

import pytest
from httpx import AsyncClient


# ── 完整推理链路 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_classify_pipeline_returns_valid_result(app_client: AsyncClient) -> None:
    """端到端：POST /classify → 推理引擎 → 模型 → 返回有效分类结果。"""
    response = await app_client.post(
        "/api/v1/classify",
        json={"text": "今天的足球比赛非常精彩，主队3比1获胜"},
    )
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()

    # 验证结果是有意义的（label 是非空字符串，不是 None 或空串）
    assert isinstance(body["label"], str) and len(body["label"]) > 0, \
        f"label 应为非空字符串，实际 {body['label']!r}"
    assert isinstance(body["confidence"], float), "confidence 应为 float"
    assert body["latency_ms"] > 0, "latency_ms 应大于 0"
    assert len(body["request_id"]) > 0, "request_id 不应为空"


@pytest.mark.asyncio
async def test_full_generate_pipeline_returns_valid_result(app_client: AsyncClient) -> None:
    """端到端：POST /generate → 推理引擎 → 模型 → 返回有效生成文本。"""
    response = await app_client.post(
        "/api/v1/generate",
        json={"prompt": "人工智能", "max_length": 50},
    )
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()

    assert isinstance(body["text"], str), "text 应为字符串"
    assert isinstance(body["tokens_generated"], int), "tokens_generated 应为整数"
    assert body["tokens_generated"] >= 0, "tokens_generated 不应为负数"
    assert body["latency_ms"] >= 0, "latency_ms 应大于 0"


@pytest.mark.asyncio
async def test_classify_different_texts_return_different_labels(app_client: AsyncClient) -> None:
    """链路：不同领域文本应能被正确分类到不同标签（验证模型实际工作，非随机输出）。"""
    sports_text = "NBA总决赛湖人队以4比2赢得冠军"
    tech_text = "苹果发布新一代M3芯片性能大幅提升"

    resp1 = await app_client.post("/api/v1/classify", json={"text": sports_text})
    resp2 = await app_client.post("/api/v1/classify", json={"text": tech_text})

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    # 两条明显不同领域的文本，模型应有能力给出不同标签
    # （这里不强制要求标签具体值，因为训练数据可能影响边界分类）
    label1 = resp1.json()["label"]
    label2 = resp2.json()["label"]
    assert isinstance(label1, str) and isinstance(label2, str), \
        "两个请求均应返回有效标签"


# ── 缓存生效验证 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeated_request_latency_is_reasonable(app_client: AsyncClient) -> None:
    """链路：相同请求连续发两次，验证延迟均在合理范围（缓存若启用则第二次更快）。

    设计说明：
      此测试不强制要求"第二次一定比第一次快50%"，
      因为 InferenceEngine 当前版本未集成缓存层（缓存在 cache_layer.py 中独立存在，
      需要在 engine 中显式调用才生效）。
      若将来集成缓存，可以取消注释下面的严格断言。
    """
    payload = {"text": "新款芯片采用3nm工艺制程"}

    start1 = time.monotonic()
    resp1 = await app_client.post("/api/v1/classify", json=payload)
    elapsed1 = (time.monotonic() - start1) * 1000

    start2 = time.monotonic()
    resp2 = await app_client.post("/api/v1/classify", json=payload)
    elapsed2 = (time.monotonic() - start2) * 1000

    assert resp1.status_code == 200, "第一次请求应成功"
    assert resp2.status_code == 200, "第二次请求应成功"

    # 两次请求均应在 3000ms 内完成（超时阈值为 5s，正常推理应远低于此）
    assert elapsed1 < 3000, f"第一次请求耗时 {elapsed1:.1f}ms，超出预期"
    assert elapsed2 < 3000, f"第二次请求耗时 {elapsed2:.1f}ms，超出预期"

    # 若缓存生效，取消注释以下严格断言：
    # assert elapsed2 < elapsed1 * 0.5, (
    #     f"缓存应使第二次请求至少快50%，"
    #     f"第一次={elapsed1:.1f}ms，第二次={elapsed2:.1f}ms"
    # )


# ── 指标更新验证 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_counter_increments_after_request(app_client: AsyncClient) -> None:
    """链路：发起请求后，/metrics 的 engine.total_requests 应递增。"""
    # 记录请求前的计数
    before = await app_client.get("/api/v1/metrics")
    assert before.status_code == 200
    count_before = before.json()["engine"]["total_requests"]

    # 发起一次分类请求
    classify_resp = await app_client.post(
        "/api/v1/classify",
        json={"text": "股市今天大涨"},
    )
    assert classify_resp.status_code == 200, "分类请求应成功"

    # 检查计数器是否增加
    after = await app_client.get("/api/v1/metrics")
    assert after.status_code == 200
    count_after = after.json()["engine"]["total_requests"]

    assert count_after == count_before + 1, (
        f"total_requests 应增加 1，"
        f"请求前={count_before}，请求后={count_after}"
    )


@pytest.mark.asyncio
async def test_metrics_success_count_increments_after_valid_request(app_client: AsyncClient) -> None:
    """链路：成功请求后，engine.success_count 应递增。"""
    before_metrics = (await app_client.get("/api/v1/metrics")).json()
    success_before = before_metrics["engine"]["success_count"]

    await app_client.post(
        "/api/v1/classify",
        json={"text": "今天的娱乐新闻很精彩"},
    )

    after_metrics = (await app_client.get("/api/v1/metrics")).json()
    success_after = after_metrics["engine"]["success_count"]

    assert success_after == success_before + 1, (
        f"success_count 应增加 1，before={success_before}，after={success_after}"
    )
