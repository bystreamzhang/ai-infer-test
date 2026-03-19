"""
测试配置文件：定义所有共享 fixtures。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixture Scope 速查
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scope 控制 fixture 的生命周期，即"多少个测试共享同一个实例"：

  session  — 整个测试会话只创建一次（最贵的操作用这个）
             适合：加载 ML 模型（TextClassifier 训练需要几百毫秒）
  module   — 每个测试文件创建一次
             适合：FastAPI AsyncClient（需要启动 lifespan）
  function — 每个测试函数独立创建（默认）
             适合：cache、rate_limiter 等有状态对象，防止测试间相互污染

为什么 cache/rate_limiter 用 function scope？
  - LRUTTLCache 有 _store、_total_requests 等可变状态
  - 若共享，test_A 写入的缓存条目会被 test_B 读到，导致测试顺序依赖
  - function scope 保证每个测试都是干净的初始状态

为什么 model 用 session scope？
  - TextClassifier.__init__ 会训练 sklearn Pipeline（~200ms）
  - 80 个单元测试 × 200ms = 16 秒，完全不必要
  - 模型本身是无状态的（predict 不修改 self），共享安全
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import app, lifespan
from src.models.text_classifier import TextClassifier
from src.models.text_generator import TextGenerator
from src.services.cache_layer import LRUTTLCache
from src.middleware.rate_limiter import TokenBucketRateLimiter, _default_limiter


# ── 模型 fixtures（session scope，只加载一次）─────────────────────────────────


@pytest.fixture(scope="session")
def classifier_model() -> TextClassifier:
    """预训练的 TextClassifier 实例，整个测试会话共享。"""
    return TextClassifier()


@pytest.fixture(scope="session")
def generator_model() -> TextGenerator:
    """预构建的 TextGenerator 实例，整个测试会话共享。"""
    return TextGenerator()


# ── HTTP 客户端 fixture（module scope）────────────────────────────────────────


@pytest_asyncio.fixture(scope="module")
async def app_client() -> AsyncClient:
    """FastAPI AsyncClient，每个测试模块共享。

    手动调用 lifespan 触发 startup（ASGITransport 不自动触发 lifespan 事件）。
    提高限流上限避免集成测试触发 429。
    """
    # 集成测试期间提高限流上限，避免高并发测试触发 429
    # 限流逻辑本身由 test_rate_limiter.py 单元测试覆盖，这里不需要测它
    _default_limiter._capacity = 10000
    _default_limiter._buckets.clear()

    async with lifespan(app):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            yield client

    # 恢复原始配置
    _default_limiter._capacity = 20
    _default_limiter._buckets.clear()



# ── 有状态组件 fixtures（function scope，每测试独立）─────────────────────────


@pytest.fixture(scope="function")
def cache_instance() -> LRUTTLCache:
    """每个测试独立的 LRUTTLCache 实例（max_size=3, ttl=60s）。"""
    return LRUTTLCache(max_size=3, ttl_seconds=60.0)


@pytest.fixture(scope="function")
def rate_limiter() -> TokenBucketRateLimiter:
    """每个测试独立的 TokenBucketRateLimiter 实例（capacity=5, refill=0.5/s）。"""
    return TokenBucketRateLimiter(capacity=5, refill_rate=0.5)


# ── 参数化数据 fixture ────────────────────────────────────────────────────────


@pytest.fixture
def sample_texts() -> list[str]:
    """用于参数化测试的典型文本列表，覆盖四个分类类别。"""
    return [
        "中国队赢得世界杯小组赛首胜",          # sports
        "人工智能大模型发布引发行业震动",        # tech
        "国产电影票房突破百亿创历史",            # entertainment
        "A股市场成交量创年内新高",              # finance
    ]
