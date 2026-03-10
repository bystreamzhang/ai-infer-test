# CLAUDE.md — AI Model Inference Testing Framework

## 项目简介

一个针对模拟AI推理服务的自动化测试框架。被测服务用FastAPI+sklearn构建，测试体系覆盖单元/集成/性能/混沌四层，集成pdb、cProfile、tracemalloc等debug工具。

## 教学模式（重要）

本项目的使用者是一名CS硕士生，有ACM竞赛背景和系统编程经验，但对FastAPI/sklearn/pytest等Python生态工具不熟悉。请严格遵循以下教学流程：

### 对于 src/ 下的每个文件，按三步走：

**第一步：前置知识讲解**
在写任何代码之前，先用1-2段话讲清楚该文件涉及的核心概念和API。例如：
- 如果要写FastAPI路由，先解释 `@app.post` 装饰器、Pydantic模型校验、依赖注入的基本原理
- 如果要用sklearn pipeline，先解释 TF-IDF 向量化和朴素贝叶斯的工作流程
- 如果要用asyncio.Queue，先解释Python异步编程模型和async/await语法

讲解要求：**只讲当前文件需要的知识，不要发散**。用类比帮助理解（用户有C++多线程经验，可以类比）。

**第二步：提供骨架代码**
给出完整的文件结构，包含：
- 所有import语句（写完整）
- 类和方法签名（含完整type hints和docstring）
- 关键逻辑用 `# TODO:` 注释标注，说明需要实现什么
- 简单的胶水代码和配置已经写好（不需要用户填的部分）
- 对于有难度的TODO，在注释中给出1-2行伪代码提示

骨架代码示例风格：
```python
def predict(self, text: str) -> dict:
    """对输入文本进行分类预测。

    Args:
        text: 待分类的文本

    Returns:
        包含 label, confidence, latency_ms 的字典
    """
    start_time = time.time()

    # TODO: 用 self.pipeline.predict_proba([text]) 获取各类别概率
    # 提示: predict_proba 返回 shape (1, n_classes) 的 numpy array
    probabilities = ...

    # TODO: 找到最大概率对应的类别标签
    # 提示: self.pipeline.classes_ 是标签数组，用 argmax 找索引
    label = ...
    confidence = ...

    elapsed_ms = (time.time() - start_time) * 1000
    return {"label": label, "confidence": float(confidence), "latency_ms": elapsed_ms}
```

**第三步：Review用户实现**
用户填完TODO后会把代码贴回来或让你检查文件。此时：
- 检查正确性，指出bug
- 检查是否符合代码规范（type hints、docstring）
- 如果有更pythonic的写法，建议但不强制
- 确认通过后再进入下一个文件

### 对于 tests/ 下的文件：

测试代码由你直接编写完整代码，但要在每个测试文件开头用注释块解释：
- 这组测试运用了什么测试方法论（等价类划分/边界值分析/状态转换等）
- 为什么选择这种方法
- pytest的关键特性（fixture/parametrize/mark等）如何在这里使用

### 对于 debug_tools/ 下的文件：

由你编写完整代码，但每个文件都是"可运行的教程"风格——代码中穿插大量注释解释每一步在做什么、为什么这么做。

## 技术栈

### 被测服务
- FastAPI + uvicorn (Web框架)
- scikit-learn (文本分类模型: TF-IDF + MultinomialNB)
- pydantic (数据校验)
- structlog (结构化日志)

### 测试
- pytest (pytest-cov, pytest-html, pytest-asyncio, pytest-benchmark)
- httpx (异步HTTP测试客户端)
- hypothesis (属性测试)
- locust (负载测试)

### Debug工具
- pdb / ipdb (断点调试)
- cProfile + snakeviz (CPU剖析)
- tracemalloc (内存追踪)
- py-spy (火焰图)

## 项目结构

```
ai-infer-test/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── src/
│   ├── app.py                    # FastAPI主入口
│   ├── models/
│   │   ├── text_classifier.py    # 文本分类（sklearn）
│   │   └── text_generator.py     # 文本生成（Markov Chain）
│   ├── services/
│   │   ├── inference_engine.py   # 推理引擎（异步队列+并发控制）
│   │   ├── model_registry.py     # 模型注册与版本管理
│   │   └── cache_layer.py        # LRU+TTL缓存
│   ├── middleware/
│   │   ├── rate_limiter.py       # 令牌桶限流
│   │   └── request_validator.py  # 请求校验
│   └── utils/
│       ├── metrics.py            # 指标采集
│       └── logger.py             # 日志配置
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   ├── performance/
│   ├── chaos/
│   └── data_driven/
├── debug_tools/
├── scripts/
├── reports/
└── docs/
    ├── phases/                   # 各阶段详细规格（供Claude Code阅读）
    ├── test_plan.md
    ├── test_strategy.md
    └── debug_guide.md
```

## 代码规范

1. 所有函数必须有完整type hints
2. 所有公共方法必须有Google Style docstring
3. 测试命名: `test_<功能>_<场景>_<期望>`，如 `test_classify_empty_input_returns_error`
4. 所有assert带描述性message
5. 使用structlog，不用print
6. 不使用magic number

## 约束

- Python 3.11+，不依赖外部API，不需要GPU
- 测试可离线运行，60秒内完成（性能测试除外）
- 覆盖率目标: src/ >= 90%

## 构建顺序

按 docs/phases/ 下的文档顺序推进。每阶段详细规格见对应文件：
1. `docs/phases/phase1_service.md` — 被测服务（教学模式：骨架填空）
2. `docs/phases/phase2_tests.md` — 测试体系（Claude直接编写）
3. `docs/phases/phase3_debug.md` — Debug工具（教程风格编写）
4. `docs/phases/phase4_docs.md` — 文档与CI脚本