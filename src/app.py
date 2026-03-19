"""FastAPI 主应用入口：把所有模块串联成一个可运行的 HTTP 服务。

本文件的职责：
1. 定义 lifespan（启动时初始化模型/引擎，关闭时清理）
2. 注册所有 API 路由
3. 挂载中间件（限流）
4. 注册全局异常处理器

不要在这里写业务逻辑——业务逻辑在 models/ 和 services/ 里，
这里只做"接线"。
"""

import uuid
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from src.middleware.rate_limiter import RateLimitMiddleware
from src.middleware.request_validator import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
)
from src.models.text_classifier import TextClassifier
from src.models.text_generator import TextGenerator
from src.services.inference_engine import InferenceEngine, InferenceRequest
from src.services.model_registry import ModelNotFoundError, ModelRegistry
from src.utils.logger import get_logger
from src.utils.metrics import registry as metrics_registry

logger = get_logger(__name__)

# ── 全局单例（在 lifespan 中初始化，通过 Depends 注入到路由） ─────────────────

_model_registry: ModelRegistry | None = None
_inference_engine: InferenceEngine | None = None


# ── Lifespan：服务启动与关闭 ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """管理应用的启动和关闭流程。

    启动时：
    1. 初始化 ModelRegistry
    2. 实例化 TextClassifier 和 TextGenerator（含模型训练）
    3. 将模型注册到 registry
    4. 初始化 InferenceEngine

    关闭时：
    记录关闭日志（本项目无需额外清理资源）。
    """
    global _model_registry, _inference_engine

    logger.info("app_starting")

    # TODO: 初始化 ModelRegistry
    # 提示：直接 _model_registry = ModelRegistry()
    _model_registry = ModelRegistry()

    # TODO: 实例化 TextClassifier，注册到 _model_registry
    # 提示：
    #   classifier = TextClassifier()
    #   _model_registry.register("text_classifier", "v1.0", classifier)
    classifier = TextClassifier()
    _model_registry.register("text_classifier", "v1.0", classifier)

    # TODO: 实例化 TextGenerator，注册到 _model_registry
    generator = TextGenerator()
    _model_registry.register("text_generator", "v1.0", generator)

    # TODO: 初始化 InferenceEngine，传入 _model_registry
    # 提示：InferenceEngine(registry=_model_registry, max_concurrency=4, timeout=5.0)
    _inference_engine = InferenceEngine(registry=_model_registry, max_concurrency=4, timeout=5.0)

    logger.info("app_started", models=_model_registry.list_models())

    yield  # ← 服务在此运行，处理所有请求

    logger.info("app_shutting_down")


# ── 创建 FastAPI 实例 ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Inference Service",
    description="模拟AI推理服务：文本分类 + 文本生成",
    version="1.0.0",
    lifespan=lifespan,
)

# ── 挂载中间件（后注册的先执行） ──────────────────────────────────────────────

# TODO: 用 app.add_middleware(RateLimitMiddleware) 挂载限流中间件
# 提示：app.add_middleware(RateLimitMiddleware)
# 注意：不需要传 limiter 参数，RateLimitMiddleware 内部有默认实例
app.add_middleware(RateLimitMiddleware)

# ── 全局异常处理器 ─────────────────────────────────────────────────────────────


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """将 ModelNotFoundError 转换为 HTTP 404 响应。

    Args:
        request: 触发异常的 HTTP 请求（框架自动传入，此处不直接使用）。
        exc: 被捕获的 ModelNotFoundError 实例。

    Returns:
        HTTP 404 JSONResponse，body 为 ErrorResponse 格式。
    """
    # TODO: 返回 JSONResponse(status_code=404, content=ErrorResponse(...).model_dump())
    # ErrorResponse 字段：code="MODEL_NOT_FOUND", message=str(exc), detail=None
    #
    # 注意 Pydantic v2 中 .model_dump() 替代了旧版的 .dict()
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            code="MODEL_NOT_FOUND",
            message=str(exc),
            detail=None,
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理器，捕获所有未处理的异常，返回 HTTP 500。

    Args:
        request: 触发异常的 HTTP 请求。
        exc: 被捕获的异常实例。

    Returns:
        HTTP 500 JSONResponse，body 为 ErrorResponse 格式。
    """
    logger.error("unhandled_exception", exc_type=type(exc).__name__, exc_msg=str(exc))
    # TODO: 返回 JSONResponse(status_code=500, ...)
    # ErrorResponse 字段：code="INTERNAL_ERROR", message="服务内部错误", detail=str(exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code="INTERNAL_ERROR",
            message="服务内部错误",
            detail=str(exc),
        ).model_dump(),
    )


# ── 依赖注入工厂函数 ───────────────────────────────────────────────────────────


def get_registry() -> ModelRegistry:
    """依赖注入：提供全局 ModelRegistry 实例。

    Returns:
        ModelRegistry 实例（在 lifespan 中初始化）。

    Raises:
        RuntimeError: 在 lifespan 初始化前调用时（正常不应发生）。
    """
    if _model_registry is None:
        raise RuntimeError("ModelRegistry not initialized")
    return _model_registry


def get_engine() -> InferenceEngine:
    """依赖注入：提供全局 InferenceEngine 实例。

    Returns:
        InferenceEngine 实例（在 lifespan 中初始化）。

    Raises:
        RuntimeError: 在 lifespan 初始化前调用时（正常不应发生）。
    """
    if _inference_engine is None:
        raise RuntimeError("InferenceEngine not initialized")
    return _inference_engine


# ── 路由：健康检查 ─────────────────────────────────────────────────────────────


@app.get("/api/v1/health", summary="健康检查")
async def health() -> dict:
    """返回服务健康状态。

    Returns:
        包含 status 字段的字典，正常时 status="ok"。
    """
    # TODO: 返回 {"status": "ok", "models_loaded": len(_model_registry.list_models())}
    # 提示：需要处理 _model_registry 为 None 的情况（服务尚未启动时）
    return {
        "status": "ok",
        "models_loaded": len(_model_registry.list_models()) if _model_registry else 0,
    }


# ── 路由：模型列表 ─────────────────────────────────────────────────────────────


@app.get("/api/v1/models", summary="列出已注册模型")
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """返回所有已注册模型的元数据。

    Args:
        registry: 通过依赖注入获取的 ModelRegistry 实例。

    Returns:
        包含 models 列表和 total 数量的字典。
    """
    # TODO: 调用 registry.list_models()，返回 {"models": [...], "total": len(...)}
    models = registry.list_models()
    return {
        "models": models,
        "total": len(models),
    }

# ── 路由：单条文本分类 ────────────────────────────────────────────────────────


@app.post(
    "/api/v1/classify",
    response_model=ClassifyResponse,
    summary="单条文本分类",
)
async def classify(
    request: ClassifyRequest,
    engine: InferenceEngine = Depends(get_engine),
) -> ClassifyResponse:
    """对单条文本进行分类。

    Args:
        request: 包含待分类文本的请求体（Pydantic 自动校验）。
        engine: 通过依赖注入获取的 InferenceEngine 实例。

    Returns:
        ClassifyResponse，包含 label、confidence、latency_ms、request_id。

    Raises:
        ModelNotFoundError: 当 text_classifier 模型未注册时（由异常处理器转为 404）。
    """
    request_id = str(uuid.uuid4())

    # TODO: 构造 InferenceRequest，调用 engine.submit()
    # 提示：
    #   infer_req = InferenceRequest(
    #       model_name="text_classifier",
    #       input_data=request.text,
    #       request_id=request_id,
    #   )
    #   response = await engine.submit(infer_req)
    infer_req = InferenceRequest(
        model_name="text_classifier",
        input_data=request.text,
        request_id=request_id,
    )
    response = await engine.submit(infer_req)  # TODO: await engine.submit(infer_req)

    # TODO: 检查 response.success，若失败则抛出 RuntimeError(response.error)
    # 提示：if not response.success: raise RuntimeError(response.error)
    if not response.success:
        raise RuntimeError(response.error)

    # TODO: 从 response.result 中提取字段，构造并返回 ClassifyResponse
    # response.result 是 TextClassifier.predict() 的返回值：
    #   {"label": str, "confidence": float, "latency_ms": float}
    result = response.result
    return ClassifyResponse(
        label=result["label"],
        confidence=result["confidence"],
        latency_ms=response.latency_ms,   # TODO: 用 response.latency_ms（端到端延迟）
        request_id=request_id,
    )


# ── 路由：批量文本分类 ────────────────────────────────────────────────────────


@app.post(
    "/api/v1/batch/classify",
    response_model=BatchClassifyResponse,
    summary="批量文本分类",
)
async def batch_classify(
    request: BatchClassifyRequest,
    engine: InferenceEngine = Depends(get_engine),
) -> BatchClassifyResponse:
    """对批量文本进行分类（逐条提交到推理引擎）。

    Args:
        request: 包含文本列表的请求体。
        engine: 通过依赖注入获取的 InferenceEngine 实例。

    Returns:
        BatchClassifyResponse，包含每条文本的分类结果列表。

    Note:
        每条文本作为独立的 InferenceRequest 提交，享受各自的超时保护。
        若某条失败，整个请求失败（抛出异常）。
    """
    start = time.monotonic()

    # TODO: 遍历 request.texts，对每条文本构造 InferenceRequest 并 await engine.submit()
    # 注意：每条文本的 request_id 应该独立生成（str(uuid.uuid4())）
    # 提示（伪代码）：
    #   results = []
    #   for text in request.texts:
    #       rid = str(uuid.uuid4())
    #       infer_req = InferenceRequest(model_name="text_classifier", input_data=text, request_id=rid)
    #       resp = await engine.submit(infer_req)
    #       if not resp.success: raise RuntimeError(resp.error)
    #       results.append(ClassifyResponse(
    #           label=resp.result["label"],
    #           confidence=resp.result["confidence"],
    #           latency_ms=resp.latency_ms,
    #           request_id=rid,
    #       ))
    results: list[ClassifyResponse] = []
    for text in request.texts:
        rid = str(uuid.uuid4())  # TODO
        infer_req = InferenceRequest(model_name="text_classifier", input_data=text, request_id=rid)
        resp = await engine.submit(infer_req)
        if not resp.success:
            raise RuntimeError(resp.error)
        results.append(ClassifyResponse(
            label=resp.result["label"],
            confidence=resp.result["confidence"],
            latency_ms=resp.latency_ms,
            request_id=rid,
        ))

    total_latency = (time.monotonic() - start) * 1000
    return BatchClassifyResponse(results=results, total=len(results), latency_ms=total_latency)


# ── 路由：文本生成 ─────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/generate",
    response_model=GenerateResponse,
    summary="文本生成",
)
async def generate(
    request: GenerateRequest,
    engine: InferenceEngine = Depends(get_engine),
) -> GenerateResponse:
    """根据 prompt 生成文本。

    Args:
        request: 包含 prompt 和 max_length 的请求体。
        engine: 通过依赖注入获取的 InferenceEngine 实例。

    Returns:
        GenerateResponse，包含生成的文本、字符数、延迟和 request_id。
    """
    request_id = str(uuid.uuid4())

    # TODO: 构造 InferenceRequest，注意：
    # - model_name="text_generator"
    # - input_data 需要同时传入 prompt 和 max_length
    #
    # 问题：InferenceRequest.input_data 是 Any 类型，
    # TextGenerator.predict(text) 只接受一个字符串参数。
    # 怎么同时传 prompt 和 max_length？
    #
    # 有三种思路：
    #   思路A：input_data = request.prompt（忽略 max_length，用模型默认值）
    #   思路B：input_data = {"prompt": request.prompt, "max_length": request.max_length}
    #          然后在 _run_inference 里特殊处理 dict 类型（但那是另一个文件，不动它）
    #   思路C：input_data = request.prompt（只传 prompt），
    #          接受 max_length 被忽略的局限，这是本项目的简化设计
    #
    # 本项目选择思路C（简化设计），即只传 prompt：
    infer_req = InferenceRequest(
        model_name="text_generator",
        input_data=request.prompt,
        request_id=request_id,
    )

    # TODO: await engine.submit(infer_req)，检查成功，提取结果
    # response.result 是 TextGenerator.predict() 的返回值：
    #   {"text": str, "tokens_generated": int, "latency_ms": float}
    response = await engine.submit(infer_req)  # TODO

    if not response.success:
        raise RuntimeError(response.error)

    result = response.result
    return GenerateResponse(
        text=result["text"],              # TODO: result["text"]
        tokens_generated=result["tokens_generated"],  # TODO: result["tokens_generated"]
        latency_ms=response.latency_ms,
        request_id=request_id,
    )


# ── 路由：指标 ─────────────────────────────────────────────────────────────────


@app.get("/api/v1/metrics", summary="查看运行时指标")
async def get_metrics(
    engine: InferenceEngine = Depends(get_engine),
) -> dict:
    """返回服务的运行时指标快照。

    Args:
        engine: 通过依赖注入获取的 InferenceEngine 实例。

    Returns:
        包含推理引擎统计和各指标快照的字典。
    """
    # TODO: 合并两个来源的指标并返回：
    # 1. engine.stats()：推理引擎的统计（total_requests, avg_latency_ms 等）
    # 2. metrics_registry.snapshot()：Counter/Gauge/Histogram 的快照
    #
    # 提示：返回 {"engine": engine.stats(), "metrics": metrics_registry.snapshot()}
    return {
        "engine": engine.stats(),
        "metrics": metrics_registry.snapshot(),
    }
