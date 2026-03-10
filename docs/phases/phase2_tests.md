# 阶段二：测试体系构建

> 本阶段由Claude直接编写完整测试代码，但每个文件开头必须用注释块解释测试方法论。
> 前置条件：阶段一的被测服务已经能正常运行。

## 构建顺序

1. `tests/conftest.py` — fixtures
2. `tests/unit/` — 全部5个文件
3. `tests/integration/` — 全部3个文件
4. `tests/data_driven/` — 2个文件 + 测试数据JSON
5. `tests/performance/test_latency_benchmark.py` — 延迟基准（MVP版只做这一个）
6. `tests/chaos/test_fault_injection.py` — 故障注入（MVP版只做这一个）

## conftest.py

提供以下fixtures：
- `app_client`（scope=module）：FastAPI AsyncClient
- `classifier_model`（scope=session）：预加载的TextClassifier实例
- `generator_model`（scope=session）：预加载的TextGenerator实例
- `cache_instance`（scope=function）：每个测试独立的LRUTTLCache实例
- `rate_limiter`（scope=function）：每个测试独立的限流器
- `sample_texts`：参数化用的测试文本列表

在文件开头注释中解释：fixture scope的含义（session/module/function），为什么cache和rate_limiter用function scope（测试隔离），为什么model用session scope（加载成本高）。

## 单元测试

### `test_text_classifier.py`
注释要解释的方法论：**等价类划分**（正常文本/空串/超长/特殊字符是不同等价类）

测试用例：
- 正常分类（每个类别至少1条已知样本，验证标签正确）
- 空字符串 → 不崩溃，返回合理结果或抛出明确异常
- 超长文本（10000+字符）→ 正常处理
- 特殊字符/emoji/HTML标签 → 不崩溃
- 返回值结构校验（必须有label, confidence, latency_ms三个key）
- confidence范围校验（0到1之间）
- batch预测结果数量与输入数量一致
- 使用 `@pytest.mark.parametrize` 参数化

### `test_text_generator.py`
注释要解释的方法论：**边界值分析**（max_length在0/1/100/500/501处的行为）

测试用例：
- 正常生成（prompt + max_length=50）
- max_length边界：0 → 空或异常，1 → 单字符，500 → 正常，501 → 正常（但触发隐藏bug的条件）
- 空prompt处理
- 返回值结构校验
- tokens_generated <= max_length

### `test_cache_layer.py`
注释要解释的方法论：**状态转换测试**（缓存条目的生命周期：不存在→存在→过期→淘汰）

测试用例：
- 基本get/put（存入后能取出，值一致）
- 未命中返回None
- LRU淘汰：容量为3时存入4个，第1个被淘汰
- TTL过期：设TTL=0.1秒，sleep(0.2)后get返回None
- 访问刷新顺序：get操作应将条目移到最新
- stats正确性：请求数、命中数、命中率计算
- 线程安全：10个线程同时读写不崩溃（用threading.Thread）
- invalidate后get返回None

### `test_rate_limiter.py`
注释要解释的方法论：**时序相关测试** + mock时间

测试用例：
- capacity=5时前5次allow返回True
- 第6次返回False
- mock时间推进后令牌恢复，再次允许
- 不同client_id独立计算
- 使用 `unittest.mock.patch('time.monotonic')` 控制时间

### `test_model_registry.py`
测试用例：
- 注册→获取→验证是同一实例
- 获取不存在的模型→ModelNotFoundError
- 多版本注册→latest指向最新
- 注销→再获取应失败
- list_models返回正确列表

## 集成测试

### `test_api_endpoints.py`
注释要解释：**接口测试的基本原则**（正向/反向/边界）

测试用例：
- `POST /api/v1/classify` 正常请求 → 200 + 正确JSON格式
- 缺失text字段 → 422
- `POST /api/v1/generate` 正常请求 → 200
- max_length=0 → 合理处理（400或空结果）
- `POST /api/v1/batch/classify` 空列表 → 422或400
- `GET /api/v1/health` → 200 + status: "healthy"
- `GET /api/v1/metrics` → 200 + 包含counter/histogram数据

### `test_inference_pipeline.py`
注释要解释：**端到端测试** vs 集成测试的区别

测试用例：
- 完整推理链路：发请求→返回正确分类结果
- 缓存生效：相同请求第二次延迟显著降低（至少快50%）
- 指标更新：请求后metrics中的counter应增加

### `test_concurrent_inference.py`
注释要解释：**并发测试设计**，如何验证并发控制和无死锁

测试用例：
- asyncio.gather 发起50个并发分类请求
- 所有请求都返回成功结果（无异常、无死锁）
- 验证耗时合理（并发4，50个请求，单请求50ms → 总时间约650ms左右）
- 统计P95延迟

## 数据驱动测试

### 测试数据文件
`tests/data_driven/test_data/normal_inputs.json`:
```json
[
  {"text": "今天的足球比赛非常精彩，主队3比1获胜", "expected_category": "体育"},
  {"text": "新款芯片采用3nm工艺制程", "expected_category": "科技"},
  ...每类5条
]
```

`tests/data_driven/test_data/edge_cases.json`:
```json
[
  {"text": "", "description": "空字符串"},
  {"text": "a", "description": "单字符"},
  {"text": "<script>alert('xss')</script>", "description": "XSS payload"},
  {"text": "' OR 1=1 --", "description": "SQL注入"},
  {"text": "🎉🎊🎈", "description": "纯emoji"},
  ...
]
```

`tests/data_driven/test_data/malicious_inputs.json`:
```json
[
  {"text": "A".repeat(100000), "description": "超长重复字符"},
  {"text": "\x00\x01\x02", "description": "控制字符"},
  ...
]
```

### `test_equivalence_partition.py`
注释解释：等价类划分方法——把输入空间划分为若干等价类，每类取一个代表测试

- 从JSON加载数据，用 `@pytest.mark.parametrize` 驱动
- 有效类：normal_inputs中的每条
- 无效类：edge_cases和malicious_inputs中的每条
- 对有效类验证分类正确，对无效类验证不崩溃且返回合理响应

### `test_boundary_values.py`
注释解释：边界值分析——在等价类的边界处重点测试（on/off-by-one）

- 输入长度：0, 1, 9999, 10000, 10001
- batch_size：0, 1, 99, 100, 101
- max_length：0, 1, 499, 500, 501

## 性能测试（MVP: 只做1个）

### `test_latency_benchmark.py`
- 使用 `pytest-benchmark` 的 `benchmark` fixture
- 测量单次分类的延迟分布
- 输出P50/P95/P99
- 设置回归阈值：P99 < 200ms

## 混沌测试（MVP: 只做1个）

### `test_fault_injection.py`
注释解释：故障注入测试的目的——验证系统在异常条件下的行为

- 用 `unittest.mock.patch` mock TextClassifier.predict 抛出 RuntimeError
- 验证API返回500而非进程崩溃
- 验证错误被记录到日志（用caplog fixture捕获）
- mock模型加载失败，验证/health返回unhealthy

## 阶段完成标准

- [ ] `pytest tests/unit/ -v` 全部通过
- [ ] `pytest tests/integration/ -v` 全部通过
- [ ] `pytest tests/data_driven/ -v` 全部通过
- [ ] `pytest tests/ --cov=src --cov-report=term-missing` 覆盖率 >= 90%
