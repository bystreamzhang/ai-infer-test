"""请求校验模型：用 Pydantic 声明所有 API 端点的入参约束。

核心思想：把"数据合不合法"这件事交给 Pydantic 声明式处理，
路由函数里只管业务逻辑，不写一行 if len(text) > xxx 这样的守卫代码。
FastAPI 检测到校验失败时自动返回 422 Unprocessable Entity。
"""

from pydantic import BaseModel, Field, field_validator


# ── 请求模型 ──────────────────────────────────────────────────────────────────


class ClassifyRequest(BaseModel):
    """单条文本分类请求。

    Attributes:
        text: 待分类的文本，长度限制 1-10000 字符。
    """

    # TODO: 声明 text 字段，类型 str，用 Field 限制 min_length=1, max_length=10000
    # 同时加上 description="待分类的文本"（会显示在 Swagger 文档里）
    text: str = Field(min_length=1, max_length=10000, description="待分类的文本")


class GenerateRequest(BaseModel):
    """文本生成请求。

    Attributes:
        prompt: 生成的起始提示词，长度限制 1-1000 字符。
        max_length: 生成的最大字符数，范围 1-1000，默认 100。
    """

    # TODO: 声明 prompt 字段，类型 str，min_length=1, max_length=1000
    prompt: str = Field(min_length=1, max_length=1000, description="生成的起始提示词")

    # TODO: 声明 max_length 字段，类型 int，默认值 100，范围 ge=1, le=1000
    max_length: int = Field(default=100, ge=1, le=1000, description="生成的最大字符数")


class BatchClassifyRequest(BaseModel):
    """批量文本分类请求。

    Attributes:
        texts: 待分类的文本列表，列表长度限制 1-100 条。
               每条文本长度限制 1-10000 字符（用 field_validator 校验）。
    """

    # TODO: 声明 texts 字段，类型 list[str]
    # 用 Field 限制列表长度：min_length=1, max_length=100
    # （注意：Field 的 min_length/max_length 作用在列表上，表示列表元素个数）
    texts: list[str] = Field(min_length=1, max_length=100, description="待分类的文本列表")

    # TODO: 用 @field_validator("texts") 校验每条文本长度不超过 10000 字符
    # 模板：
    #   @field_validator("texts")
    #   @classmethod
    #   def check_each_text_length(cls, v: list[str]) -> list[str]:
    #       for text in v:
    #           if len(text) > 10000:
    #               raise ValueError(...)
    #       return v
    @field_validator("texts")
    @classmethod
    def check_each_text_length(cls, v: list[str]) -> list[str]:
        for text in v:
            if len(text) > 10000:
                raise ValueError("每条文本的长度不能超过 10000 个字符")
        return v


# ── 响应模型 ──────────────────────────────────────────────────────────────────


class ClassifyResponse(BaseModel):
    """单条文本分类响应。

    Attributes:
        label: 预测的类别标签，如 "体育"、"科技"。
        confidence: 预测置信度，范围 [0.0, 1.0]。
        latency_ms: 推理耗时（毫秒）。
        request_id: 请求唯一标识符，用于日志追踪。
    """

    label: str
    confidence: float
    latency_ms: float
    request_id: str


class GenerateResponse(BaseModel):
    """文本生成响应。

    Attributes:
        text: 生成的文本内容。
        tokens_generated: 生成的字符数。
        latency_ms: 推理耗时（毫秒）。
        request_id: 请求唯一标识符。
    """

    text: str
    tokens_generated: int
    latency_ms: float
    request_id: str


class BatchClassifyResponse(BaseModel):
    """批量文本分类响应。

    Attributes:
        results: 每条文本对应的分类结果列表，顺序与请求一致。
        total: 本次批量请求的文本总数。
        latency_ms: 整批推理的总耗时（毫秒）。
    """

    results: list[ClassifyResponse]
    total: int
    latency_ms: float


# ── 统一错误响应 ──────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """统一错误响应格式，所有异常处理器都返回此结构。

    Attributes:
        code: 机器可读的错误码，如 "MODEL_NOT_FOUND"、"RATE_LIMIT_EXCEEDED"。
        message: 人类可读的错误描述。
        detail: 可选的详细信息（如校验失败的字段名）。
    """

    code: str
    message: str
    detail: str | None = None
