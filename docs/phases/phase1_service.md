# 阶段一：被测服务搭建

> 本阶段采用教学模式：对每个文件先讲前置知识，再给骨架代码让用户填空，最后review。

## 构建顺序

按以下顺序逐文件推进，每完成一个文件确认能import无报错后再继续：

1. `src/utils/logger.py`
2. `src/models/text_classifier.py`
3. `src/models/text_generator.py`
4. `src/services/cache_layer.py`
5. `src/services/model_registry.py`
6. `src/services/inference_engine.py`
7. `src/middleware/rate_limiter.py`
8. `src/middleware/request_validator.py`
9. `src/utils/metrics.py`
10. `src/app.py`

先创建 `pyproject.toml` / `requirements.txt` 和所有 `__init__.py`。

## 各文件详细规格

### 1. `src/utils/logger.py` — 结构化日志

**前置知识要点**：structlog的配置方式、processor chain概念、如何绑定上下文。

**需要实现**：
- 配置 structlog，输出JSON格式日志
- 提供 `get_logger(name: str)` 工厂函数
- 配置 processor：添加时间戳、日志级别、调用位置

**TODO难度**：低（配置为主，3个TODO左右）

---

### 2. `src/models/text_classifier.py` — 文本分类模型

**前置知识要点**：
- sklearn Pipeline 是什么：把多个处理步骤串起来（类比Unix管道）
- TF-IDF 向量化：把文本变成数值向量（讲清楚TF和IDF各自的含义）
- MultinomialNB：朴素贝叶斯分类器（讲到"假设特征独立"即可）
- `fit` vs `predict` vs `predict_proba` 的区别

**需要实现**：
- 类 `TextClassifier`，`__init__` 中构建pipeline并用内置数据训练
- 内置训练数据：4个类别（体育/科技/娱乐/财经），每类20条中文短文本
- `predict(text: str) -> dict`：返回 `{"label", "confidence", "latency_ms"}`
- `predict_batch(texts: list[str]) -> list[dict]`
- 模拟真实延迟：`time.sleep(random.uniform(0.01, 0.05))`

**TODO清单**（骨架中需要用户填写的部分）：
1. 构建 sklearn Pipeline（TfidfVectorizer + MultinomialNB）
2. 调用 pipeline.fit(texts, labels) 训练
3. predict中用 predict_proba 获取概率，用 argmax 找最大概率的类别
4. predict_batch 中循环或用pipeline的批量predict

**TODO难度**：中（涉及numpy数组操作，给出shape提示）

---

### 3. `src/models/text_generator.py` — 文本生成（Markov Chain）

**前置知识要点**：
- Markov Chain 文本生成原理：基于前N个字预测下一个字的概率分布
- Python的 `collections.defaultdict` 和 `random.choices`
- **重要**：这里需要故意埋入一个内存泄漏bug

**需要实现**：
- 类 `TextGenerator`，`__init__` 中用内置中文语料构建转移概率表
- `build_chain(corpus: str, order: int = 2) -> dict`：构建N-gram转移表
- `generate(prompt: str, max_length: int = 100) -> dict`
- 返回 `{"text", "tokens_generated", "latency_ms"}`

**故意埋入的bug**（在骨架代码的注释中用隐蔽方式引导）：
```python
# generate方法内部
self._history.append(generated_text)  # 看似用于调试记录
# 但没有任何清理机制，当 max_length > 500 时生成大量文本不断累积
# 后续 debug_tools/memory_tracker.py 会用 tracemalloc 定位到这行
```
在骨架中把这行写好（不作为TODO），让它看起来像正常代码。用户在阶段三用tracemalloc时会发现它。

**TODO清单**：
1. build_chain：遍历语料构建n-gram到下一字符的映射
2. generate：从prompt末尾的n-gram开始，逐字符采样生成
3. 长度控制和终止条件

**TODO难度**：中高（Markov Chain逻辑需要理解，给详细伪代码提示）

---

### 4. `src/services/cache_layer.py` — LRU + TTL 缓存

**前置知识要点**：
- LRU缓存原理（用 `collections.OrderedDict` 实现，`move_to_end` 方法）
- TTL（Time-To-Live）：每条数据带过期时间戳
- 线程安全：`threading.Lock` 的使用（类比C++的 `std::mutex`）
- 为什么不直接用 `functools.lru_cache`：我们需要TTL和统计

**需要实现**：
- 类 `LRUTTLCache(max_size: int, ttl_seconds: float)`
- `get(key: str) -> Optional[Any]`：命中返回值，未命中/过期返回None
- `put(key: str, value: Any) -> None`：存入，超过max_size时淘汰最久未用
- `invalidate(key: str) -> bool`
- `stats() -> dict`：返回 `{"hit_rate", "total_requests", "evictions"}`
- 所有公共方法需要加锁

**TODO清单**：
1. get中检查是否过期（比较timestamp），过期则删除
2. get命中时用 `move_to_end` 更新访问顺序
3. put中检查容量，超限时用 `popitem(last=False)` 淘汰最旧
4. stats中计算命中率（注意除零保护）
5. 用 `threading.Lock` 保护临界区

**TODO难度**：中（用户有C++多线程经验，Lock类比mutex讲解即可）

---

### 5. `src/services/model_registry.py` — 模型注册中心

**前置知识要点**：
- 注册表模式（Registry Pattern）
- Python dataclass 或 TypedDict 用于模型元数据

**需要实现**：
- 类 `ModelRegistry`
- `register(name: str, version: str, model_instance: Any) -> None`
- `get(name: str, version: str = "latest") -> Any`
- `unregister(name: str, version: str) -> None`
- `list_models() -> list[dict]`
- 版本管理：同名模型多版本，"latest"指向最新注册的

**TODO清单**：
1. 内部存储结构设计（嵌套dict: `{name: {version: ModelInfo}}`）
2. register时更新latest指针
3. get找不到时抛出自定义 `ModelNotFoundError`

**TODO难度**：低（纯Python数据结构操作）

---

### 6. `src/services/inference_engine.py` — 异步推理引擎

**前置知识要点**：
- Python asyncio 事件循环（类比C++的event loop，但是协作式调度）
- `asyncio.Queue`：异步安全的任务队列
- `asyncio.Semaphore`：控制并发数（类比C++ `std::counting_semaphore`）
- `asyncio.wait_for`：给协程加超时
- async/await 语法的本质：遇到await时让出执行权

**需要实现**：
- 类 `InferenceEngine(max_concurrency: int = 4, timeout: float = 3.0)`
- `async submit(request: InferenceRequest) -> InferenceResponse`
- 内部用Semaphore控制同时推理的并发数
- 超时机制：`asyncio.wait_for` 包裹推理调用
- 统计指标：total_requests, success_count, failure_count, avg_latency

**TODO清单**：
1. 用 `asyncio.Semaphore` 限制并发
2. 用 `asyncio.wait_for` 实现超时
3. 捕获 `asyncio.TimeoutError` 并记录
4. 更新统计指标（注意async环境下的计数安全性）

**TODO难度**：中高（async是新概念，需要详细类比C++线程模型讲解）

---

### 7. `src/middleware/rate_limiter.py` — 令牌桶限流

**前置知识要点**：
- 令牌桶算法原理：桶以固定速率填充令牌，每个请求消耗一个令牌，桶满则丢弃新令牌
- `time.monotonic()` 用于计时（为什么不用 `time.time()`）

**需要实现**：
- 类 `TokenBucketRateLimiter(capacity: int, refill_rate: float)`
- `allow(client_id: str) -> bool`：判断是否允许请求
- 按client_id独立维护桶
- FastAPI中间件集成方式（在骨架中提供中间件包装代码）

**TODO清单**：
1. 计算自上次请求以来应该补充多少令牌
2. 更新令牌数（不超过capacity）
3. 判断令牌是否足够，不够则拒绝

**TODO难度**：低中（算法简单，关键在理解时间差计算）

---

### 8. `src/middleware/request_validator.py` — 请求校验

**前置知识要点**：
- Pydantic BaseModel 定义请求/响应 schema
- FastAPI 如何自动做校验（422响应）

**需要实现**：
- `ClassifyRequest(text: str)` 带长度校验（1-10000字符）
- `GenerateRequest(prompt: str, max_length: int = 100)` 带范围校验
- `BatchClassifyRequest(texts: list[str])` 带列表长度校验（1-100条）
- 统一错误响应格式 `ErrorResponse(code, message, detail)`

**TODO难度**：低（Pydantic声明式，基本是填字段和validator）

---

### 9. `src/utils/metrics.py` — 指标采集

**前置知识要点**：
- Prometheus指标类型：Counter, Gauge, Histogram
- 这里不引入prometheus_client，自己实现轻量版

**需要实现**：
- `Counter`：只增不减的计数器
- `Histogram`：记录值的分布，能算P50/P95/P99
- `MetricsRegistry`：全局注册表，暴露为dict格式

**TODO清单**：
1. Histogram 的 observe 方法：存入值列表
2. Histogram 的 percentile 计算（排序后取位置）

**TODO难度**：低（纯数学计算）

---

### 10. `src/app.py` — FastAPI主应用

**前置知识要点**：
- FastAPI app实例创建、路由注册
- 依赖注入 (`Depends`)
- 中间件注册顺序
- Lifespan事件（启动时加载模型）

**需要实现**：
- API端点：`/api/v1/classify`, `/api/v1/generate`, `/api/v1/batch/classify`, `/api/v1/models`, `/api/v1/health`, `/api/v1/metrics`
- 启动时初始化模型、注册到registry
- 挂载限流中间件
- 统一异常处理器

**TODO清单**：
1. 定义各路由函数，调用对应service
2. lifespan中初始化模型和引擎
3. 异常处理器捕获自定义异常返回对应HTTP状态码

**TODO难度**：中（FastAPI整合所有前面的模块，是串联的关键）

---

## 阶段完成标准

- [ ] `uvicorn src.app:app --reload` 能正常启动
- [ ] 用 curl/httpx 手动测试每个端点都有正确响应
- [ ] `ruff check src/` 无报错
- [ ] `mypy src/ --ignore-missing-imports` 无报错
