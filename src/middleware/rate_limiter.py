"""令牌桶限流中间件。

核心思想：用懒惰计算代替后台线程——不维护定时任务，
而是每次请求到来时根据时间差一次性补充应有的令牌。
这样既省资源，又天然线程安全（同一 client_id 的计算是独立的）。
"""

import threading
import time
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── 单个客户端的桶状态 ────────────────────────────────────────────────────────


@dataclass
class _BucketState:
    """记录某个 client_id 的令牌桶当前状态。

    Attributes:
        tokens: 当前桶中的令牌数（浮点数，允许小数累积）。
        last_refill_time: 上次补充令牌时的时间戳（来自 time.monotonic()）。
    """

    tokens: float
    last_refill_time: float = field(default_factory=time.monotonic)


# ── 令牌桶核心逻辑 ────────────────────────────────────────────────────────────


class TokenBucketRateLimiter:
    """基于令牌桶算法的限流器，按 client_id 独立维护各自的桶。

    使用方式：
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=2.0)
        if limiter.allow("user_123"):
            # 处理请求
        else:
            # 返回 429

    Args:
        capacity: 桶的最大容量（令牌上限），也是允许的瞬间突发请求数。
        refill_rate: 每秒向桶中补充的令牌数。
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        # 每个 client_id 对应一个 _BucketState
        self._buckets: dict[str, _BucketState] = {}
        # 多线程场景下保护 _buckets 字典的锁（FastAPI 同步中间件中会用到）
        self._lock = threading.Lock()

        logger.info(
            "rate_limiter_initialized",
            capacity=capacity,
            refill_rate=refill_rate,
        )

    def allow(self, client_id: str) -> bool:
        """判断 client_id 的本次请求是否被允许。

        内部流程：
        1. 取出（或初始化）该 client_id 的桶状态
        2. 根据时间差计算应补充的令牌数
        3. 更新令牌数，不超过 capacity
        4. 若令牌数 >= 1，消耗一个令牌并返回 True；否则返回 False

        Args:
            client_id: 客户端标识符，通常是 IP 地址或 API Key。

        Returns:
            True 表示允许请求，False 表示应被限流（429）。
        """
        with self._lock:
            now = time.monotonic()

            # 如果是新客户端，初始化为满桶
            if client_id not in self._buckets:
                self._buckets[client_id] = _BucketState(
                    tokens=float(self._capacity),
                    last_refill_time=now    # ← 直接传入已经算好的 now，不调用default_factory，否则影响mock测试中的时间控制
                )

            bucket = self._buckets[client_id]

            # TODO: 计算自上次补充以来经过的秒数
            # 提示：elapsed = now - bucket.last_refill_time
            elapsed: float = now - bucket.last_refill_time

            # TODO: 计算应补充的令牌数，并更新桶的令牌数（不超过 capacity）
            # 提示：新令牌 = elapsed * self._refill_rate
            #       bucket.tokens = min(self._capacity, bucket.tokens + 新令牌)
            # 注意：不管是否允许请求，都要先补充令牌、更新时间戳
            #
            # 下面三种写法，只有一种正确，请选择：
            #
            # 写法A：
            #   bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
            #   bucket.last_refill_time = now
            #
            # 写法B（时间戳只在允许时更新）：
            #   bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
            #   if bucket.tokens >= 1:
            #       bucket.last_refill_time = now
            #
            # 写法C（先更新时间戳再补充）：
            #   bucket.last_refill_time = now
            #   bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
            #   # （elapsed 仍是旧值，这里没有错误）
            #
            bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._refill_rate)
            bucket.last_refill_time = now

            # TODO: 判断令牌是否充足，充足则消耗一个令牌返回 True，否则返回 False
            # 提示：if bucket.tokens >= 1.0: bucket.tokens -= 1; return True
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1
                return True
            return False

    def get_bucket_state(self, client_id: str) -> dict:
        """返回指定客户端的桶状态快照，用于调试和测试。

        Args:
            client_id: 客户端标识符。

        Returns:
            包含 tokens 和 last_refill_time 的字典；若客户端不存在则返回空桶状态。
        """
        with self._lock:
            if client_id not in self._buckets:
                return {"tokens": float(self._capacity), "last_refill_time": None}
            bucket = self._buckets[client_id]
            return {
                "tokens": bucket.tokens,
                "last_refill_time": bucket.last_refill_time,
            }

    def reset(self, client_id: str) -> None:
        """清除指定客户端的桶状态（令牌数恢复为满桶）。

        用于测试或手动解除某个客户端的限流。

        Args:
            client_id: 要重置的客户端标识符。
        """
        with self._lock:
            self._buckets.pop(client_id, None)


# ── FastAPI 中间件包装 ─────────────────────────────────────────────────────────

# 全局限流器实例（将在 app.py 中被挂载）
# capacity=20：允许瞬间突发 20 个请求
# refill_rate=10.0：每秒补充 10 个令牌（稳态 QPS 上限）
_default_limiter = TokenBucketRateLimiter(capacity=20, refill_rate=10.0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件：在每个请求进入路由前检查限流。

    从请求头 X-Client-ID 读取客户端标识，缺省时回退到真实IP。
    被限流的请求返回 HTTP 429，并在响应头中说明原因。

    Args:
        app: FastAPI/Starlette 应用实例（由框架自动传入）。
        limiter: TokenBucketRateLimiter 实例，缺省使用模块级全局实例。
    """

    def __init__(
        self,
        app,
        limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter or _default_limiter

    async def dispatch(self, request: Request, call_next):
        """拦截请求，检查限流后决定放行或拒绝。

        Args:
            request: 当前 HTTP 请求对象。
            call_next: 调用后续中间件/路由的回调。

        Returns:
            放行时返回后续处理的响应；限流时返回 429 JSONResponse。
        """
        # 优先从自定义 header 读 client_id，没有则用 IP
        client_id = request.headers.get("X-Client-ID") or (
            request.client.host if request.client else "unknown"
        )

        if not self._limiter.allow(client_id):
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "请求过于频繁，请稍后重试",
                    "detail": f"客户端 {client_id} 超过请求速率限制",
                },
            )

        response = await call_next(request)
        return response
