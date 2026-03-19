"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
集成测试：API 端点测试
测试方法论：接口测试的正向/反向/边界三原则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【接口测试的核心思想】
单元测试验证"一个函数做对了吗"，接口测试验证"HTTP请求-响应链路是否完整且正确"。
两者的区别在于：接口测试真正走完了 JSON 解析 → 路由分发 → 业务逻辑 → 序列化 这整条链路。

三原则：
  正向测试（Happy Path）：合法输入，验证成功响应的格式和内容是否符合规范
  反向测试（Negative Path）：非法输入，验证错误响应的 HTTP 状态码和格式
  边界测试（Boundary）：临界值，验证系统在极端输入下的行为（而非崩溃）

【pytest 特性说明】
  @pytest.mark.asyncio：标记测试函数为 async，需要 pytest-asyncio
  app_client：来自 conftest.py 的 module-scope AsyncClient fixture，
               module scope 让同一文件的所有测试复用同一个已启动的 ASGI 应用，
               避免每个测试都重新触发 lifespan（重新训练模型）
  pytest_asyncio.ini_options asyncio_mode = "auto"：在 pyproject.toml 中配置后，
               不再需要手动打 @pytest.mark.asyncio 标记
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pytest
from httpx import AsyncClient


# ── /api/v1/classify ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_valid_request_returns_200(app_client: AsyncClient) -> None:
    """正向：正常文本分类请求应返回 200 及完整 JSON 结构。"""
    response = await app_client.post(
        "/api/v1/classify",
        json={"text": "中国队赢得世界杯小组赛首胜"},
    )
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()
    assert "label" in body, "响应体缺少 label 字段"
    assert "confidence" in body, "响应体缺少 confidence 字段"
    assert "latency_ms" in body, "响应体缺少 latency_ms 字段"
    assert "request_id" in body, "响应体缺少 request_id 字段"


@pytest.mark.asyncio
async def test_classify_valid_request_confidence_in_range(app_client: AsyncClient) -> None:
    """正向：返回的 confidence 应在 [0, 1] 区间内。"""
    response = await app_client.post(
        "/api/v1/classify",
        json={"text": "新款芯片采用3nm工艺制程"},
    )
    assert response.status_code == 200
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0, f"confidence={confidence} 不在 [0, 1] 范围内"


@pytest.mark.asyncio
async def test_classify_missing_text_field_returns_422(app_client: AsyncClient) -> None:
    """反向：缺少 text 字段应返回 422 Unprocessable Entity（Pydantic 校验失败）。"""
    response = await app_client.post(
        "/api/v1/classify",
        json={"wrong_field": "hello"},
    )
    assert response.status_code == 422, f"期望 422，实际 {response.status_code}"


@pytest.mark.asyncio
async def test_classify_empty_text_returns_422(app_client: AsyncClient) -> None:
    """边界：空字符串应被 Field(min_length=1) 拦截，返回 422。"""
    response = await app_client.post(
        "/api/v1/classify",
        json={"text": ""},
    )
    assert response.status_code == 422, f"期望 422，实际 {response.status_code}"


@pytest.mark.asyncio
async def test_classify_text_at_max_length_returns_200(app_client: AsyncClient) -> None:
    """边界：恰好 10000 字符的文本应被接受，返回 200。"""
    text = "测" * 10000
    response = await app_client.post("/api/v1/classify", json={"text": text})
    assert response.status_code == 200, f"10000 字符应被接受，实际 {response.status_code}"


@pytest.mark.asyncio
async def test_classify_text_over_max_length_returns_422(app_client: AsyncClient) -> None:
    """边界：10001 字符的文本应被 Field(max_length=10000) 拦截，返回 422。"""
    text = "a" * 10001
    response = await app_client.post("/api/v1/classify", json={"text": text})
    assert response.status_code == 422, f"期望 422，实际 {response.status_code}"


# ── /api/v1/generate ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_valid_request_returns_200(app_client: AsyncClient) -> None:
    """正向：正常文本生成请求应返回 200 及完整 JSON 结构。"""
    response = await app_client.post(
        "/api/v1/generate",
        json={"prompt": "今天天气", "max_length": 50},
    )
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()
    assert "text" in body, "响应体缺少 text 字段"
    assert "tokens_generated" in body, "响应体缺少 tokens_generated 字段"
    assert "latency_ms" in body, "响应体缺少 latency_ms 字段"
    assert "request_id" in body, "响应体缺少 request_id 字段"


@pytest.mark.asyncio
async def test_generate_max_length_zero_returns_422(app_client: AsyncClient) -> None:
    """边界：max_length=0 应被 Field(ge=1) 拦截，返回 422。"""
    response = await app_client.post(
        "/api/v1/generate",
        json={"prompt": "测试", "max_length": 0},
    )
    assert response.status_code == 422, f"期望 422，实际 {response.status_code}"


@pytest.mark.asyncio
async def test_generate_missing_prompt_returns_422(app_client: AsyncClient) -> None:
    """反向：缺少 prompt 字段应返回 422。"""
    response = await app_client.post(
        "/api/v1/generate",
        json={"max_length": 50},
    )
    assert response.status_code == 422, f"期望 422，实际 {response.status_code}"


# ── /api/v1/batch/classify ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_classify_valid_request_returns_200(app_client: AsyncClient) -> None:
    """正向：批量分类正常请求应返回 200，results 数量与输入一致。"""
    texts = ["中国队获胜", "人工智能突破", "票房大卖"]
    response = await app_client.post(
        "/api/v1/batch/classify",
        json={"texts": texts},
    )
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()
    assert body["total"] == len(texts), f"期望 total={len(texts)}，实际 {body['total']}"
    assert len(body["results"]) == len(texts), "results 数量与输入不符"


@pytest.mark.asyncio
async def test_batch_classify_empty_list_returns_422(app_client: AsyncClient) -> None:
    """反向：空列表应被 Field(min_length=1) 拦截，返回 422。"""
    response = await app_client.post(
        "/api/v1/batch/classify",
        json={"texts": []},
    )
    assert response.status_code == 422, f"期望 422，实际 {response.status_code}"


# ── /api/v1/health ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_200_and_ok_status(app_client: AsyncClient) -> None:
    """正向：健康检查应返回 200，且 status 为 "ok"。"""
    response = await app_client.get("/api/v1/health")
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()
    assert body["status"] == "ok", f"期望 status='ok'，实际 '{body.get('status')}'"


@pytest.mark.asyncio
async def test_health_returns_models_loaded_count(app_client: AsyncClient) -> None:
    """正向：健康检查应返回已加载的模型数量（lifespan 注册了2个模型）。"""
    response = await app_client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "models_loaded" in body, "响应体缺少 models_loaded 字段"
    assert body["models_loaded"] >= 2, f"期望至少2个模型，实际 {body['models_loaded']}"


# ── /api/v1/metrics ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_returns_200_with_engine_and_metrics(app_client: AsyncClient) -> None:
    """正向：指标端点应返回 200，包含 engine 和 metrics 两个顶层字段。"""
    response = await app_client.get("/api/v1/metrics")
    assert response.status_code == 200, f"期望 200，实际 {response.status_code}"
    body = response.json()
    assert "engine" in body, "响应体缺少 engine 字段"
    assert "metrics" in body, "响应体缺少 metrics 字段"


@pytest.mark.asyncio
async def test_metrics_engine_has_required_fields(app_client: AsyncClient) -> None:
    """正向：engine 字段应包含 total_requests, success_count 等统计字段。"""
    response = await app_client.get("/api/v1/metrics")
    assert response.status_code == 200
    engine_stats = response.json()["engine"]
    for field in ("total_requests", "success_count", "failure_count", "avg_latency_ms"):
        assert field in engine_stats, f"engine 缺少字段 {field}"
