"""
cprofile_analysis.py — CPU 性能剖析教程
========================================

目标：用 cProfile 对 TextClassifier 做 1000 次推理，找出 CPU 热点。

运行方式：
    python debug_tools/cprofile_analysis.py

依赖：
    pip install snakeviz   # 用于可视化 .prof 文件（可选）
"""

import cProfile
import pstats
import io
import sys
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 【知识点 1】cProfile vs profile
# --------------------------------------------------------------------------- #
# Python 标准库提供两个性能剖析模块：
#
#   - profile  : 纯 Python 实现，剖析开销大（会干扰测量结果）
#   - cProfile : C 语言实现，开销小，推荐生产场景使用
#
# 两者 API 完全相同，只需把 `import profile` 换成 `import cProfile`。
# 一般规则：**永远用 cProfile**，除非你在调试剖析器本身。
# --------------------------------------------------------------------------- #

# 把项目根目录加入 sys.path，让 src/ 可以被正常导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.text_classifier import TextClassifier  # noqa: E402  (路径处理完再导入)

# --------------------------------------------------------------------------- #
# 【知识点 2】pstats 各列含义
# --------------------------------------------------------------------------- #
# 剖析输出有 6 列，含义如下：
#
#   ncalls    : 该函数被调用的总次数
#               如果是 "3/1" 表示递归调用：3 次总调用，1 次原始调用
#
#   tottime   : 函数自身执行的总时间（不含子函数调用的时间）
#               tottime 高 → 该函数自身逻辑重，是"自身瓶颈"
#
#   percall   : tottime / ncalls（每次调用的平均自身耗时）
#
#   cumtime   : 该函数及其所有子函数的累计执行时间
#               cumtime 高 → 该函数是调用链上的瓶颈（可能是因为调了很重的子函数）
#
#   percall   : cumtime / ncalls（第二个 percall，含子函数的平均耗时）
#
#   filename:lineno(function) : 函数定位信息
#
# 分析策略：
#   1. 先看 cumtime 排行榜 → 找出"调用链上最贵"的函数（从上向下优化）
#   2. 再看 tottime 排行榜 → 找出"自身最贵"的函数（最直接的优化点）
#   3. ncalls 异常高的函数也值得关注（可能是不必要的重复计算）
# --------------------------------------------------------------------------- #

# 用于剖析的测试文本（覆盖四个类别，让分类器有真实工作量）
TEST_TEXTS = [
    "人工智能大模型引发行业革命",
    "NBA总决赛湖人队夺冠",
    "A股市场成交量创新高",
    "电影票房突破百亿",
    "量子计算机突破传统极限",
    "足球运动员签约创纪录转会费",
    "央行降准释放长期流动性",
    "综艺节目收视率连续夺冠",
]

REPEAT_COUNT = 1000   # 每次推理的总次数
REPORTS_DIR = PROJECT_ROOT / "reports"
PROF_FILE = REPORTS_DIR / "inference_profile.prof"


def run_inference_workload(classifier: TextClassifier, n: int) -> None:
    """执行 n 次推理作为剖析的工作负载。

    Args:
        classifier: 已初始化的 TextClassifier 实例。
        n: 总推理次数。
    """
    # 循环调用 predict，覆盖多种输入文本，模拟真实负载
    for i in range(n):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        classifier.predict(text)


def profile_with_context_manager(classifier: TextClassifier) -> cProfile.Profile:
    """用 context manager 方式使用 cProfile（推荐写法）。

    Returns:
        包含剖析数据的 Profile 对象。
    """
    # --------------------------------------------------------------------------- #
    # 【知识点 3】cProfile.Profile() 三种使用方式
    # --------------------------------------------------------------------------- #
    # 方式一：context manager（最推荐，作用域清晰）
    #
    #   with cProfile.Profile() as pr:
    #       some_code()
    #   # 退出 with 块后，pr 中已经有完整的剖析数据
    #
    # 方式二：手动 enable/disable（适合只剖析部分代码）
    #
    #   pr = cProfile.Profile()
    #   pr.enable()
    #   some_code()
    #   pr.disable()
    #
    # 方式三：命令行直接剖析整个脚本（适合不改代码的情况）
    #
    #   python -m cProfile -o output.prof your_script.py
    #   python -m cProfile -s cumtime your_script.py   # 按 cumtime 排序输出
    # --------------------------------------------------------------------------- #

    print(f"\n{'='*60}")
    print(f"开始剖析：对 TextClassifier 执行 {REPEAT_COUNT} 次推理")
    print(f"{'='*60}\n")

    # 注意：TextClassifier.__init__ 里有 time.sleep(random.uniform(0.01, 0.05))
    # 这是故意模拟的推理延迟，因此 1000 次推理会花 10-50 秒，请耐心等待
    # 如果只想快速看结果，可以把 REPEAT_COUNT 改小（如 50）
    print(f"⚠️  注意：每次推理模拟 10~50ms 延迟，{REPEAT_COUNT} 次约需 10-50 秒...")
    print("   提示：修改 REPEAT_COUNT = 50 可快速验证输出格式\n")

    with cProfile.Profile() as pr:
        run_inference_workload(classifier, REPEAT_COUNT)

    return pr


def print_stats_by_cumtime(pr: cProfile.Profile, top_n: int = 20) -> None:
    """按 cumtime（累计时间）打印 top N 耗时函数。

    Args:
        pr: 已完成剖析的 Profile 对象。
        top_n: 展示前 N 条记录。
    """
    # --------------------------------------------------------------------------- #
    # pstats.Stats 的使用流程：
    #   1. Stats(profile, stream=buffer)  — 创建统计对象，stream 重定向输出
    #   2. .strip_dirs()                  — 去掉文件路径中冗长的目录前缀，让输出更简洁
    #   3. .sort_stats('cumulative')      — 按 cumtime 降序排序
    #   4. .print_stats(top_n)            — 打印前 top_n 条
    #
    # sort_stats 支持的排序关键字（常用）：
    #   'cumulative' / 'cumtime'  : 按累计时间（找调用链瓶颈）
    #   'tottime' / 'time'        : 按自身时间（找函数内部热点）
    #   'calls' / 'ncalls'        : 按调用次数（找频繁调用）
    #   'filename'                : 按文件名
    # --------------------------------------------------------------------------- #

    buffer = io.StringIO()
    stats = pstats.Stats(pr, stream=buffer)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)

    output = buffer.getvalue()
    print(f"\n{'='*60}")
    print(f"TOP {top_n} 耗时函数（按 cumtime 累计时间排序）")
    print("分析思路：cumtime 高 = 整条调用链上的瓶颈")
    print(f"{'='*60}")
    print(output)


def print_stats_by_tottime(pr: cProfile.Profile, top_n: int = 20) -> None:
    """按 tottime（自身时间）打印 top N 耗时函数。

    Args:
        pr: 已完成剖析的 Profile 对象。
        top_n: 展示前 N 条记录。
    """
    buffer = io.StringIO()
    stats = pstats.Stats(pr, stream=buffer)
    stats.strip_dirs()
    stats.sort_stats("tottime")
    stats.print_stats(top_n)

    output = buffer.getvalue()
    print(f"\n{'='*60}")
    print(f"TOP {top_n} 耗时函数（按 tottime 自身时间排序）")
    print("分析思路：tottime 高 = 函数自身逻辑是瓶颈，优化这里最直接")
    print(f"{'='*60}")
    print(output)


def save_prof_file(pr: cProfile.Profile) -> None:
    """将剖析数据保存为 .prof 文件，供 snakeviz 可视化。

    Args:
        pr: 已完成剖析的 Profile 对象。
    """
    # --------------------------------------------------------------------------- #
    # 【知识点 4】用 snakeviz 可视化 .prof 文件
    # --------------------------------------------------------------------------- #
    # 安装：pip install snakeviz
    #
    # 使用：snakeviz reports/inference_profile.prof
    #       → 自动在浏览器打开交互式火焰图和调用树
    #
    # snakeviz 提供两种视图：
    #   - Icicle（冰柱图）：调用树从上往下，根是最外层调用，叶是最内层
    #   - Sunburst（旭日图）：调用树以圆形展示，更直观但复杂调用树难以阅读
    #
    # 交互技巧：
    #   - 点击任意函数可以"下钻"，聚焦查看该函数的调用子树
    #   - 右上角的 "depth" 控制展示层数
    #   - 搜索框可以过滤特定函数名
    #
    # 与 cProfile 文字输出的互补关系：
    #   文字输出 → 精确数字，适合定量比较
    #   snakeviz → 调用关系可视化，适合理解"为什么"某函数 cumtime 高
    # --------------------------------------------------------------------------- #

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(str(PROF_FILE))
    print(f"\n✅ 剖析数据已保存：{PROF_FILE}")
    print(f"\n📊 可视化命令（需要先安装 snakeviz）：")
    print(f"   pip install snakeviz")
    print(f"   snakeviz {PROF_FILE}")
    print(f"\n   或使用命令行查看（无需安装 snakeviz）：")
    print(f"   python -m pstats {PROF_FILE}")


def print_key_observations(pr: cProfile.Profile) -> None:
    """从剖析数据中提取关键洞察，帮助理解输出结果。

    Args:
        pr: 已完成剖析的 Profile 对象。
    """
    # --------------------------------------------------------------------------- #
    # 【知识点 5】如何从剖析结果中识别优化机会
    # --------------------------------------------------------------------------- #
    # 1. time.sleep 占大头 → 这是故意的模拟延迟，真实推理服务不该有
    #    → 在测试/benchmark 时应该 mock 掉 time.sleep
    #
    # 2. TfidfVectorizer.transform 耗时较高 → TF-IDF 向量化是主要计算开销
    #    → 优化方向：预热缓存、批量化处理、改用更快的向量化库
    #
    # 3. MultinomialNB.predict_proba 耗时 → 相对轻量
    #    → 对于小模型，sklearn 的实现已经足够高效
    #
    # 4. Python 函数调用本身的开销（如 __call__、__getattr__）
    #    → 如果 ncalls 极大但 tottime 很小，说明是调用链分散的合理开销
    #
    # 实际项目中，看到剖析结果后的分析流程：
    #   Step 1: 找出 cumtime TOP 3，理解调用链
    #   Step 2: 看 tottime TOP 3，找可以直接优化的函数
    #   Step 3: 对高 ncalls 函数检查是否有重复计算
    #   Step 4: 写 benchmark，量化优化前后的提升
    # --------------------------------------------------------------------------- #

    # 统计总调用次数和总时间
    buffer = io.StringIO()
    stats = pstats.Stats(pr, stream=buffer)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    # pstats.Stats 内部有 total_calls 和 total_tt 属性
    total_calls = stats.total_calls
    total_time = stats.total_tt

    print(f"\n{'='*60}")
    print("关键统计摘要")
    print(f"{'='*60}")
    print(f"总推理次数   : {REPEAT_COUNT}")
    print(f"总函数调用数 : {total_calls:,}")
    print(f"总消耗时间   : {total_time:.3f} 秒")
    print(f"平均每次推理 : {total_time / REPEAT_COUNT * 1000:.2f} ms")
    print(f"\n{'='*60}")
    print("分析提示")
    print(f"{'='*60}")
    print("• time.sleep 会在 tottime 排行榜前列 → 这是故意模拟的延迟")
    print("• TfidfVectorizer.transform 是真正的计算热点")
    print("• 如需消除 sleep 干扰，可 mock：")
    print("  unittest.mock.patch('time.sleep')")


def main() -> None:
    """主函数：执行完整的 CPU 剖析流程。"""

    # Step 1：初始化分类器（训练过程不在剖析范围内）
    # 注意：TextClassifier.__init__ 会训练模型，这一步故意放在 profile 之外
    # 因为在真实推理服务中，模型只训练一次，我们只关心推理阶段的性能
    print("正在初始化 TextClassifier（训练阶段，不计入剖析）...")
    classifier = TextClassifier()
    print("✅ TextClassifier 初始化完成\n")

    # Step 2：运行剖析
    pr = profile_with_context_manager(classifier)

    # Step 3：打印关键摘要
    print_key_observations(pr)

    # Step 4：按 cumtime 排序打印 top 20（找调用链瓶颈）
    print_stats_by_cumtime(pr, top_n=20)

    # Step 5：按 tottime 排序打印 top 20（找自身逻辑热点）
    print_stats_by_tottime(pr, top_n=20)

    # Step 6：保存 .prof 文件
    save_prof_file(pr)

    print(f"\n{'='*60}")
    print("剖析完成！建议后续步骤：")
    print("  1. 用 snakeviz 可视化调用树，理解 cumtime 高的原因")
    print("  2. 重点关注 TfidfVectorizer.transform 的耗时")
    print("  3. 尝试 mock time.sleep 后重新剖析，看真实推理开销")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
