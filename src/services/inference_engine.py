"""推理引擎：异步并发控制与超时管理。

核心思想：把同步的、CPU密集型的模型推理包装进异步框架，
让服务在等待某个推理完成的同时，能继续接受并处理其他请求。
用 Semaphore 做"闸门"，防止太多请求同时冲进来打垮系统。
"""
from src.services.model_registry import ModelNotFoundError

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── 请求/响应数据结构 ────────────────────────────────────────────────────────


@dataclass
class InferenceRequest:
    """推理请求，统一封装模型名称、版本和输入数据。

    Attributes:
        model_name: 目标模型名称，如 "text_classifier"。
        input_data: 传给模型的原始输入，类型由模型决定（str 或 list[str]）。
        request_id: 唯一标识符，用于日志追踪。
        version: 请求的模型版本，默认 "latest"。
    """

    model_name: str
    input_data: Any
    request_id: str
    version: str = "latest"


@dataclass
class InferenceResponse:
    """推理响应，统一封装结果和元数据。

    Attributes:
        request_id: 与请求对应的唯一标识符。
        result: 推理结果，类型由模型决定。
        latency_ms: 本次推理的端到端耗时（毫秒），含排队等信号量的时间。
        success: 是否推理成功。
        error: 失败时的错误描述，成功时为 None。
    """

    request_id: str
    result: Any
    latency_ms: float
    success: bool
    error: str | None = None


# ── 推理引擎 ─────────────────────────────────────────────────────────────────


class InferenceEngine:
    """异步推理引擎，通过 Semaphore 控制并发、wait_for 控制超时。

    使用方式：
        engine = InferenceEngine(registry, max_concurrency=4, timeout=3.0)
        response = await engine.submit(request)

    Args:
        registry: ModelRegistry 实例，用于按名称+版本查找模型。
        max_concurrency: 同时运行的最大推理数，超过时新请求排队等待。
        timeout: 单次推理的超时秒数，超时后返回失败响应。
    """

    def __init__(
        self,
        registry: Any,
        max_concurrency: int = 4,
        timeout: float = 3.0,
    ) -> None:
        self._registry = registry
        self._timeout = timeout

        # TODO: 初始化 asyncio.Semaphore，限制最大并发数为 max_concurrency
        # 提示：asyncio.Semaphore(n) 创建初始值为n的信号量
        # async with self._semaphore: 获取一个slot，超过n个时第n+1个在这里挂起
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)  # type: ignore[assignment]

        # 统计指标（asyncio 单线程，普通 int/float 即可，无需加锁）
        self._total_requests: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._total_latency_ms: float = 0.0

        logger.info(
            "inference_engine_initialized",
            max_concurrency=max_concurrency,
            timeout=timeout,
        )

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """提交一个推理请求，返回推理结果。

        完整流程：
        1. 记录开始时间
        2. 获取 Semaphore slot（若并发已满则在此挂起等待）
        3. 在 Semaphore 保护范围内，用 wait_for 执行带超时的推理
        4. 捕获 TimeoutError，构造失败响应
        5. 更新统计指标
        6. 记录日志并返回响应

        Args:
            request: 包含模型名称、版本、输入数据的推理请求。

        Returns:
            InferenceResponse，无论成功还是超时，总是返回（不抛出异常）。
        """
        start_time = time.monotonic()

        # TODO: 用 async with self._semaphore: 包裹以下所有逻辑
        # 提示：进入 async with 后，信号量计数-1；退出后+1
        # 当计数为0时（达到最大并发），新的 async with 会挂起等待
        #
        # 在 async with 块内：
        #   try:
        #     result = await asyncio.wait_for(
        #         self._run_inference(request),
        #         timeout=self._timeout
        #     )
        #     # 构造成功的 InferenceResponse
        #   except asyncio.TimeoutError:
        #     # 构造失败的 InferenceResponse，error 字段填超时描述
        #
        # 注意：无论成功还是失败，都要在 async with 块内更新统计指标
        async with self._semaphore:
            self._total_requests += 1
            try:
                result = await asyncio.wait_for(
                    self._run_inference(request),
                    timeout=self._timeout,
                )
                latency_ms = (time.monotonic() - start_time) * 1000
                self._success_count += 1
                self._total_latency_ms += latency_ms

                response = InferenceResponse(
                    request_id=request.request_id,
                    result=result,
                    latency_ms=latency_ms,
                    success=True,
                    error=None,
                )
            except asyncio.TimeoutError:
                self._failure_count += 1
                latency_ms = (time.monotonic() - start_time) * 1000
                self._total_latency_ms += latency_ms
                response = InferenceResponse(
                    request_id=request.request_id,
                    result=None,
                    latency_ms=latency_ms,
                    success=False,
                    error=f"推理超时（超过 {self._timeout} 秒）",
                )
            except ModelNotFoundError:
                raise  # 让它穿透到 app.py 的 @exception_handler(ModelNotFoundError)
            except Exception as e:
                self._failure_count += 1
                latency_ms = (time.monotonic() - start_time) * 1000
                self._total_latency_ms += latency_ms
                response = InferenceResponse(
                    request_id=request.request_id,
                    result=None,
                    latency_ms=latency_ms,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                )
        return response


    async def _run_inference(self, request: InferenceRequest) -> Any:
        """实际执行推理的内部方法（不对外暴露）。

        从 registry 取出模型，用 asyncio.to_thread 在线程池中运行同步的 predict，
        避免阻塞事件循环。

        Args:
            request: 推理请求。

        Returns:
            模型的推理结果（dict 或 list[dict]）。

        Raises:
            ModelNotFoundError: 当模型不存在时（由 registry.get 抛出）。
            任何模型内部抛出的异常。
        """
        # 从注册中心获取模型实例
        model = self._registry.get(request.model_name, request.version)

        # TODO: 用 asyncio.to_thread 在线程池中调用模型的 predict 方法
        # 为什么需要 to_thread：predict 是同步阻塞函数（含 time.sleep），
        # 直接 await 一个普通函数是不行的，它会阻塞整个事件循环。
        # to_thread 把它丢到线程池执行，返回一个可以 await 的协程。
        #
        # 有三种写法，只有一种正确：
        #
        # 写法A（调用 predict）：
        #   return await asyncio.to_thread(model.predict, request.input_data)
        #
        # 写法B（调用 predict_batch，注意 input_data 是 list 时用这个）：
        #   return await asyncio.to_thread(model.predict_batch, request.input_data)
        #
        # 写法C（错误写法，这样会先同步调用 predict 再 await 结果）：
        #   return await asyncio.to_thread(model.predict(request.input_data))
        #
        # 这里我们统一用 predict（TextClassifier 和 TextGenerator 都有 predict 方法），
        # 请选择正确的写法：
        result = await asyncio.to_thread(model.predict, request.input_data)  # TODO: 替换为正确的 asyncio.to_thread 调用
        return result

    def stats(self) -> dict:
        """返回引擎的运行时统计指标。

        Returns:
            包含以下字段的字典：
            - total_requests: 累计提交的请求数
            - success_count: 成功完成的请求数
            - failure_count: 超时或异常的请求数
            - avg_latency_ms: 所有请求的平均延迟（毫秒），无请求时为 0.0
        """
        # TODO: 计算并返回统计字典
        # 注意：avg_latency_ms 需要除零保护（total_requests 可能为 0）
        # 伪代码：
        #   avg = total_latency / total_requests if total_requests > 0 else 0.0
        result = {
            "total_requests": self._total_requests,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "avg_latency_ms": self._total_latency_ms / self._total_requests if self._total_requests > 0 else 0.0
        }
        return result
