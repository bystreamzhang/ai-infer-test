# 阶段三：Debug工具集成

> 本阶段由Claude编写完整代码，但必须是"可运行的教程"风格：代码中穿插大量注释解释每一步。
> 每个脚本都应该能独立运行，并产生有意义的输出。

## 构建顺序

1. `debug_tools/pdb_debug_example.py`
2. `debug_tools/memory_tracker.py` （重点：发现阶段一埋入的内存泄漏bug）
3. `debug_tools/cprofile_analysis.py`
4. `debug_tools/flame_graph_gen.py`
5. `debug_tools/log_analyzer.py`

## 各文件规格

### 1. `pdb_debug_example.py` — 断点调试教程

**代码结构**：

- 导入被测服务的TextClassifier
- 在关键位置设置 `breakpoint()`（Python 3.7+的标准方式）
- 用大段注释讲解pdb命令：
  - `n`(next), `s`(step), `c`(continue)
  - `p/pp`(print/pretty-print变量)
  - `w`(where/查看调用栈), `l`(list源码)
  - `b`(设置断点), `cl`(清除断点)
  - `!expression`(执行Python表达式)
- 演示一个调试场景：分类结果不符合预期时，如何逐步检查pipeline中间结果
- 在注释中提供VS Code launch.json配置示例
- 额外提到ipdb的安装和使用（`pip install ipdb`, `PYTHONBREAKPOINT=ipdb.set_trace`）

**运行方式**：`python debug_tools/pdb_debug_example.py`（会进入交互式调试）

---

### 2. `memory_tracker.py` — 内存泄漏检测（核心演示）

**代码结构**：

- 使用 `tracemalloc` 模块
- 流程：
  1. `tracemalloc.start()` 开始追踪
  2. `snapshot1 = tracemalloc.take_snapshot()` 取基线快照
  3. 循环调用 `TextGenerator.generate(max_length=600)` 共500次
  4. `snapshot2 = tracemalloc.take_snapshot()` 取第二次快照
  5. `stats = snapshot2.compare_to(snapshot1, 'lineno')` 对比
  6. 打印 top 10 内存增长来源
- 注释中详细解释：
  - tracemalloc的工作原理（hook了Python的内存分配器）
  - snapshot对比的三种key_type：'lineno', 'filename', 'traceback'
  - 如何从输出中定位到 text_generator.py 的 `self._history.append` 行
  - 为什么这是内存泄漏（引用未释放，GC无法回收）
- 对比实验：max_length=100 跑500次 vs max_length=600 跑500次，打印内存差异
- 最后给出修复建议（限制_history长度或改用deque(maxlen=N)）

**运行方式**：`python debug_tools/memory_tracker.py`

**期望输出**：清晰显示 text_generator.py 某行的内存持续增长

---

### 3. `cprofile_analysis.py` — CPU性能剖析

**代码结构**：

- 用 `cProfile.Profile()` 作为context manager
- 剖析目标：对TextClassifier做1000次predict
- 用 `pstats.Stats` 排序（cumulative time, total time）
- 打印 top 20 耗时函数
- 保存 .prof 文件到 reports/
- 注释讲解：
  - cProfile vs profile（C实现 vs 纯Python实现，性能差异）
  - 各列含义：ncalls, tottime, percall, cumtime
  - 如何用 snakeviz 可视化：`snakeviz reports/inference_profile.prof`
  - 如何识别瓶颈：cumtime高的函数是调用链上的瓶颈，tottime高的是自身逻辑重

**运行方式**：`python debug_tools/cprofile_analysis.py`

---

### 4. `flame_graph_gen.py` — 火焰图生成

**代码结构**：

- 这个文件主要是一个启动器+说明文档
- 用 `subprocess` 启动被测服务（uvicorn）
- 生成 py-spy 命令（因为py-spy需要attach到进程）
- 同时用httpx发一些请求制造负载
- 注释讲解：
  - 火焰图怎么读：x轴是采样栈的聚合宽度（越宽=占CPU越多），y轴是调用深度
  - py-spy record 和 py-spy top 的区别
  - `py-spy record -o reports/flamegraph.svg --pid <PID>` 命令说明
  - 什么是采样式profiler（相比cProfile的插桩式，对性能影响更小）

**运行方式**：`python debug_tools/flame_graph_gen.py`（会打印py-spy命令供手动执行）

---

### 5. `log_analyzer.py` — 日志分析工具

**代码结构**：

- 读取structlog输出的JSON格式日志文件
- 统计分析：
  - 错误率（ERROR级别占比）
  - 慢请求占比（latency_ms > 200）
  - 各端点请求分布（按path分组统计）
  - 每分钟请求数时间序列
- 支持命令行参数：`--log-file`, `--time-window`（最近N分钟）
- 输出格式化的问题摘要报告
- 注释讲解：
  - 结构化日志 vs 传统文本日志的优势（可解析、可聚合）
  - 在生产环境中如何用ELK/Grafana做类似分析
  - 问题复盘的流程：定位→归因→修复→验证→预防

**运行方式**：`python debug_tools/log_analyzer.py --log-file reports/app.log`

---

## 阶段完成标准

- [X] 每个debug_tools/脚本都能独立运行无报错
- [X] memory_tracker.py 能清晰定位到 text_generator.py 的内存泄漏行
- [X] cprofile_analysis.py 输出 top 20 耗时函数且保存.prof文件
- [X] 所有脚本的注释清晰易懂，可作为独立学习材料
