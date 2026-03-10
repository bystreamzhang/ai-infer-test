# 阶段四：文档与CI脚本

> 本阶段文档建议用户自己撰写主要内容，Claude可以提供大纲和模板。
> CI脚本由Claude直接编写。

## 文档

### `docs/test_plan.md` — 测试计划

由Claude提供模板框架，以下章节标题和1-2句描述需要用户自己填充展开：

1. **测试目标**：验证AI推理服务的功能正确性、性能达标、容错能力
2. **测试范围**
   - 在范围内：分类/生成API功能、缓存策略、限流机制、并发处理、错误处理
   - 不在范围内：前端UI、部署流程、安全渗透测试
3. **测试环境**：Python 3.11+, 本地运行, 无需GPU/网络
4. **测试项与优先级矩阵**（表格：测试项 | 优先级P0-P3 | 测试类型 | 状态）
5. **进入标准**：代码通过ruff+mypy检查，服务能正常启动
6. **退出标准**：覆盖率>=90%，P0/P1用例100%通过，无阻塞性bug
7. **风险评估**（表格：风险 | 概率 | 影响 | 缓解措施）

### `docs/test_strategy.md` — 测试策略

同样提供框架，由用户展开：

1. **测试分层策略**（画出测试金字塔）
   - 单元测试（60%）：模型、缓存、限流器的独立功能
   - 集成测试（25%）：API端点、推理流水线、并发
   - 性能测试（10%）：延迟基准、吞吐量
   - 混沌测试（5%）：故障注入、资源耗尽
2. **测试用例设计方法**
   - 等价类划分：如何应用于本项目（举具体例子）
   - 边界值分析：如何应用于本项目
   - 状态转换：缓存条目的生命周期
3. **自动化策略**：全部测试自动化，CI中分层执行（快的先跑）
4. **缺陷管理**：severity定义、报告模板

### `docs/debug_guide.md` — 调试指南

由Claude编写完整内容（因为这是工具使用教程性质）：

1. pdb/ipdb 快速上手
2. cProfile + snakeviz 排查性能问题流程
3. tracemalloc 排查内存泄漏流程（以本项目的bug为实例）
4. py-spy 火焰图分析
5. structlog 日志分析
6. 常见问题排查checklist

### `README.md`

由Claude编写，包含：
- 项目简介（一段话）
- 快速开始（安装依赖、启动服务、运行测试 三条命令）
- 项目结构说明（树形图+一句话描述每个目录）
- 测试运行命令（分层运行 + 覆盖率 + 报告生成）
- 项目亮点（用bullet list，对标JD关键词）

## CI脚本

### `scripts/run_all_tests.sh`

```bash
#!/bin/bash
set -e

echo "=== Linting ==="
ruff check src/ tests/

echo "=== Type checking ==="
mypy src/ --ignore-missing-imports

echo "=== Unit tests ==="
pytest tests/unit/ -v --tb=short

echo "=== Integration tests ==="
pytest tests/integration/ -v --tb=short

echo "=== Data-driven tests ==="
pytest tests/data_driven/ -v --tb=short

echo "=== Performance benchmark ==="
pytest tests/performance/test_latency_benchmark.py --benchmark-only --benchmark-sort=mean

echo "=== Chaos tests ==="
pytest tests/chaos/ -v --tb=short

echo "=== Coverage report ==="
pytest tests/ --cov=src --cov-report=html:reports/coverage_html --cov-report=term-missing --cov-fail-under=90

echo "=== All passed ==="
```

### `scripts/generate_report.py`

收集各测试结果，生成聚合HTML报告到 `reports/summary.html`：
- 测试通过率（从pytest-html报告解析）
- 覆盖率数字
- 性能benchmark结果（从benchmark JSON解析）
- 生成时间戳

### `Makefile`

```makefile
.PHONY: install test lint typecheck serve report clean

install:
	pip install -r requirements.txt --break-system-packages

serve:
	uvicorn src.app:app --reload --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

report:
	pytest tests/ --cov=src --cov-report=html:reports/coverage_html --html=reports/test_report.html
	python scripts/generate_report.py

clean:
	rm -rf reports/* .pytest_cache .mypy_cache __pycache__
```

## 阶段完成标准

- [ ] README.md 完整可读
- [ ] `make test` 全部通过
- [ ] `make report` 生成HTML报告
- [ ] docs/ 下三个文档结构完整
