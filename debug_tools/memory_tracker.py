"""memory_tracker.py — 用 tracemalloc 检测内存泄漏

【教程目标】
    演示如何用 Python 内置的 tracemalloc 模块定位内存泄漏，
    并通过对比实验证明 text_generator.py 中存在的真实 bug。

【运行方式】
    python debug_tools/memory_tracker.py

【期望结果】
    清晰显示 text_generator.py 第128行 self._history.append(result_text)
    的内存持续增长，且 max_length=600 时增长量约为 max_length=100 的6倍。
"""

# ============================================================
# 0. 导入
# ============================================================
import tracemalloc        # Python 3.4+ 内置，无需安装
import linecache          # 用于读取源码行，供 tracemalloc 格式化输出
import sys
import os
from collections import deque  # 后面"修复方案"章节会用到

# 把项目根目录加入 sys.path，使 "from src.xxx" 的导入生效
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.text_generator import TextGenerator  # noqa: E402


# ============================================================
# 1. tracemalloc 原理简介
# ============================================================
#
# tracemalloc 通过 **hook**（钩子）Python 的底层内存分配器 PyMalloc 工作：
#   - 当 Python 分配一块内存时，tracemalloc 记录：分配的字节数 + 调用栈
#   - 当 Python 释放内存时，tracemalloc 同步删除对应记录
#   - snapshot() 就是把当前所有"仍存活的内存块"拍一张快照
#
# 因此 snapshot2.compare_to(snapshot1) 的结果是：
#   "在 snapshot1 到 snapshot2 之间，哪些代码行净增了多少内存"
#
# 三种 key_type（分组维度）：
#   'lineno'    : 按文件名+行号分组（最精确，推荐）
#   'filename'  : 按文件名分组（粒度粗一些）
#   'traceback' : 按完整调用栈分组（信息最多，但输出也最多）


def print_top_stats(stats: list, title: str, top_n: int = 10) -> None:
    """格式化打印内存统计的 top N 条目。

    Args:
        stats: tracemalloc.compare_to() 返回的 StatisticDiff 列表（已排序）。
        title: 报告标题。
        top_n: 显示条数。
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"{'排名':<4} {'文件:行号':<55} {'净增内存':>12} {'分配次数':>10}")
    print("-" * 85)

    for rank, stat in enumerate(stats[:top_n], start=1):
        # stat.traceback[0] 是最直接的分配来源（调用栈顶）
        frame = stat.traceback[0]

        # 截断过长的文件路径，只显示从项目根目录开始的相对路径
        filename = frame.filename
        try:
            filename = os.path.relpath(filename, ROOT)
        except ValueError:
            pass  # Windows 上跨盘符时 relpath 会抛异常，忽略即可

        location = f"{filename}:{frame.lineno}"

        # size_diff 是净增字节数（正数=增长，负数=释放）
        size_kb = stat.size_diff / 1024
        sign = "+" if size_kb >= 0 else ""
        print(f"  #{rank:<3} {location:<55} {sign}{size_kb:>8.1f} KB  {stat.count_diff:>+8}")

    print()


# ============================================================
# 2. 核心实验：单次运行500次调用，对比前后快照
# ============================================================

def run_single_experiment(max_length: int, n_calls: int = 500) -> tuple[int, list]:
    """运行一次内存泄漏检测实验。

    流程：
        1. start()    ← 打开 tracemalloc 监控
        2. snapshot1  ← 基线快照（生成器初始化后）
        3. 循环调用   ← 制造内存压力
        4. snapshot2  ← 对比快照
        5. compare    ← 找出净增最多的代码行

    Args:
        max_length: 每次 generate() 调用的最大生成长度。
        n_calls: 调用次数。

    Returns:
        (总净增字节数, top 统计列表)
    """
    print(f"\n[实验] max_length={max_length}, 调用 {n_calls} 次...")

    # ── 步骤 1：启动 tracemalloc ──────────────────────────────
    # nframe=5 表示保留调用栈最深 5 层，方便定位间接调用
    tracemalloc.start(5)

    # ── 步骤 2：初始化被测对象，然后取基线快照 ────────────────
    generator = TextGenerator(order=2)

    # 取快照1：此时内存是"干净"的初始状态
    snapshot1 = tracemalloc.take_snapshot()

    # ── 步骤 3：循环调用，制造内存压力 ───────────────────────
    for i in range(n_calls):
        generator.generate(prompt="人工智能", max_length=max_length)

        # 每100次打印一次进度，避免用户以为程序卡住了
        if (i + 1) % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"  进度: {i+1}/{n_calls}  当前追踪内存: {current/1024:.1f} KB  峰值: {peak/1024:.1f} KB")

    # ── 步骤 4：取对比快照 ────────────────────────────────────
    snapshot2 = tracemalloc.take_snapshot()

    # ── 步骤 5：对比两次快照 ──────────────────────────────────
    #
    # compare_to 第二个参数是 key_type，决定如何分组：
    #   'lineno'  → 按"文件:行号"聚合（我们想精确到行，所以用这个）
    #
    # 返回值是 StatisticDiff 列表，每个元素包含：
    #   .traceback  : 调用栈（列表，[0] 是最直接的分配点）
    #   .size_diff  : 净增字节数
    #   .count_diff : 净增分配次数
    stats = snapshot2.compare_to(snapshot1, "lineno")

    # compare_to 已经按 size_diff 降序排列，直接取 top
    total_growth = sum(s.size_diff for s in stats if s.size_diff > 0)

    tracemalloc.stop()

    return total_growth, stats


# ============================================================
# 3. 对比实验：max_length=100 vs max_length=600
# ============================================================

def run_comparison_experiment() -> None:
    """对比不同 max_length 下的内存增长，验证泄漏量与生成长度成正比。

    如果 _history 没有上界，那么每次 append 进去的字符串长度正比于 max_length，
    因此 max_length=600 的实验内存增长应约为 max_length=100 的 6 倍。
    这是一个非常有力的"泄漏是 _history.append 导致"的证据。
    """
    print("\n" + "=" * 60)
    print("  对比实验：max_length=100 vs max_length=600")
    print("=" * 60)

    growth_100, stats_100 = run_single_experiment(max_length=100, n_calls=500)
    print_top_stats(stats_100, "max_length=100, 500次调用 — Top 10 内存增长", top_n=10)

    growth_600, stats_600 = run_single_experiment(max_length=600, n_calls=500)
    print_top_stats(stats_600, "max_length=600, 500次调用 — Top 10 内存增长", top_n=10)

    print("=" * 60)
    print("  内存增长对比汇总")
    print("=" * 60)
    print(f"  max_length=100 → 总净增: {growth_100 / 1024:.1f} KB")
    print(f"  max_length=600 → 总净增: {growth_600 / 1024:.1f} KB")

    if growth_100 > 0:
        ratio = growth_600 / growth_100
        print(f"  增长倍数: {ratio:.1f}x  （理论预期 ≈ 6x）")
        print()
        if ratio > 3:
            print("  ✓ 确认：内存增长量与 max_length 成正比，符合 _history 无界增长的特征")
        else:
            print("  ? 比值偏低，可能受 GC 时机影响，可尝试增大 n_calls 重现")
    print()


# ============================================================
# 4. 精确定位泄漏行：traceback 模式
# ============================================================

def locate_leak_with_traceback() -> None:
    """用 'traceback' 模式打印完整调用链，精确定位泄漏根因。

    'lineno' 模式告诉你"哪行在增长"，
    'traceback' 模式告诉你"谁调用了那行"——对间接调用尤为有用。
    """
    print("\n" + "=" * 60)
    print("  精确定位：使用 traceback 模式")
    print("=" * 60)

    # nframe=10 保留更深的调用栈
    tracemalloc.start(10)

    generator = TextGenerator(order=2)
    snapshot1 = tracemalloc.take_snapshot()

    for _ in range(200):
        generator.generate(prompt="深度学习", max_length=400)

    snapshot2 = tracemalloc.take_snapshot()

    # 用 'traceback' 分组，可以看到完整的调用链
    stats = snapshot2.compare_to(snapshot1, "traceback")

    print("\n  Top 3 内存增长来源（含完整调用栈）：\n")
    for i, stat in enumerate(stats[:3], 1):
        print(f"  ── #{i} 净增 {stat.size_diff/1024:.1f} KB ──")
        # stat.traceback 是 Traceback 对象，可以格式化为字符串列表
        for line in stat.traceback.format():
            # 去掉多余空行，缩进对齐
            for subline in line.splitlines():
                if subline.strip():
                    print(f"    {subline}")
        print()

    tracemalloc.stop()


# ============================================================
# 5. 修复方案说明
# ============================================================

def show_fix_suggestion() -> None:
    """打印修复建议。

    不实际修改 text_generator.py，只打印说明，供学习参考。
    """
    print("=" * 60)
    print("  修复建议")
    print("=" * 60)
    print("""
  【问题根因】
    text_generator.py 第128行：
        self._history.append(result_text)

    每次 generate() 调用都把完整的生成文本追加进 self._history，
    但从未删除。随着调用次数增加，_history 列表持续增长，
    其中所有字符串对象都有 _history 持有的引用，GC 无法回收。

    这是典型的"意外引用持有（Unintended Reference Retention）"模式。

  【修复方案 A：有界 deque（推荐）】
    把 list 换成 collections.deque(maxlen=N)，
    deque 在 maxlen 满时自动丢弃最旧的元素：

        from collections import deque
        self._history: deque[str] = deque(maxlen=100)  # 只保留最近100条

    优点：O(1) append，自动淘汰，代码改动最小。

  【修复方案 B：直接删除 _history】
    如果 _history 只是调试用途，生产代码里直接删掉这行即可。
    用 structlog 的日志替代历史记录：
        logger.debug("generate done", text_preview=result_text[:50])

  【修复后验证】
    重新运行本脚本，应看到 _history.append 那行消失在 top 10 之外，
    且两次实验的内存增长倍数接近 1x（不再与 max_length 成正比）。
""")


# ============================================================
# 6. 主入口
# ============================================================

def main() -> None:
    """主函数：依次运行对比实验、精确定位和修复建议。"""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          memory_tracker.py — 内存泄漏检测教程            ║")
    print("║  工具：tracemalloc（Python 3.4+ 内置，无需安装）          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 实验一：对比不同 max_length，证明泄漏量与生成长度成正比
    run_comparison_experiment()

    # 实验二：用 traceback 模式打印完整调用链，精确到文件+行号
    locate_leak_with_traceback()

    # 打印修复建议
    show_fix_suggestion()

    print("=" * 60)
    print("  分析完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
