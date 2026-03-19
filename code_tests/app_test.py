"""
app.py 冒烟测试（smoke test）

测试方法论：
- 这不是完整的测试套件，而是"能跑起来吗"的快速验证
- 覆盖每个端点的 happy path（正常输入 → 期望格式的响应）
- 用 httpx.AsyncClient + ASGITransport 直接在进程内调用 FastAPI，
  不需要真正启动 uvicorn 服务器

运行方式：
    python -m pytest code_tests/app_test.py -v
"""

import pytest
import httpx
from httpx import AsyncClient, ASGITransport

from src.app import app


# ── fixture：共享的异步 HTTP 客户端 ──────────────────────────────────────────
# lifespan=True 让 pytest 触发 app 的 lifespan（即加载模型），
# 否则路由里的 get_engine() 会抛 RuntimeError

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def client():
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as c:
            yield c


# ── 健康检查 ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] == 2   # text_classifier + text_generator


# ── 模型列表 ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_list_models(client: AsyncClient):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "models" in data
    assert "total" in data
    assert data["total"] == 2
    names = [m["name"] for m in data["models"]]
    assert "text_classifier" in names
    assert "text_generator" in names


# ── 单条文本分类 ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_classify_sports(client: AsyncClient):
    resp = await client.post(
        "/api/v1/classify",
        json={"text": "NBA总决赛湖人队夺冠"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["label"] in ("sports", "tech", "entertainment", "finance")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] > 0
    assert "request_id" in data


@pytest.mark.anyio
async def test_classify_empty_text_returns_422(client: AsyncClient):
    """空字符串应被 Pydantic 拒绝，返回 422。"""
    resp = await client.post("/api/v1/classify", json={"text": ""})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_classify_missing_field_returns_422(client: AsyncClient):
    """缺少 text 字段应返回 422。"""
    resp = await client.post("/api/v1/classify", json={})
    assert resp.status_code == 422


# ── 批量文本分类 ──────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_batch_classify(client: AsyncClient):
    resp = await client.post(
        "/api/v1/batch/classify",
        json={"texts": ["篮球比赛精彩绝伦", "人工智能改变世界", "股市今日大涨"]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3
    assert data["latency_ms"] > 0
    for item in data["results"]:
        assert "label" in item
        assert "confidence" in item
        assert "request_id" in item


@pytest.mark.anyio
async def test_batch_classify_empty_list_returns_422(client: AsyncClient):
    """空列表应被 Pydantic 拒绝，返回 422。"""
    resp = await client.post("/api/v1/batch/classify", json={"texts": []})
    assert resp.status_code == 422


# ── 文本生成 ──────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_generate(client: AsyncClient):
    resp = await client.post(
        "/api/v1/generate",
        json={"prompt": "今天天气", "max_length": 50},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data["text"], str)
    assert len(data["text"]) > 0
    assert data["tokens_generated"] >= 0
    assert data["latency_ms"] >= 0
    assert "request_id" in data


@pytest.mark.anyio
async def test_generate_empty_prompt_returns_422(client: AsyncClient):
    resp = await client.post("/api/v1/generate", json={"prompt": ""})
    assert resp.status_code == 422


# ── 指标 ─────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_metrics_after_requests(client: AsyncClient):
    """在前面的测试跑完之后，引擎应该记录到 > 0 的请求数。"""
    resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "engine" in data
    assert "metrics" in data
    assert data["engine"]["total_requests"] > 0


# ── 限流（快速发送超过桶容量的请求） ─────────────────────────────────────────

@pytest.mark.anyio
async def test_rate_limit_triggers_429(client: AsyncClient):
    """连续发送 50 个轻量请求，应触发至少一次 429。"""
    statuses = []
    for _ in range(50):
        r = await client.post(
            "/api/v1/health",
            headers={"X-Client-ID": "test-ratelimit-client"},
        )
        statuses.append(r.status_code)
    assert 429 in statuses, f"Expected at least one 429, got: {set(statuses)}"
