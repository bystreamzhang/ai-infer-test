"""
pdb_debug_example.py — 断点调试教程（可运行版）
================================================

这个脚本演示如何用 pdb 调试 TextClassifier 的预测过程。
运行后会进入交互式调试会话，你可以亲手检查 pipeline 的中间结果。

运行方式：
    python debug_tools/pdb_debug_example.py

退出调试：输入 q 或 Ctrl+D（不会报错，属于正常退出）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【PDB 核心命令速查表】

  导航命令（控制执行流）：
    n (next)      ── 执行下一行（不进入被调用函数内部）
    s (step)      ── 执行下一行（会进入被调用函数内部）
    c (continue)  ── 继续运行直到下一个断点或程序结束
    r (return)    ── 运行到当前函数的 return 语句
    q (quit)      ── 强制退出调试器（会抛出 BdbQuit）

  查看命令（检查状态）：
    p  <expr>     ── 打印表达式的值，如 p label
    pp <expr>     ── 美化打印（pretty-print），适合打印字典/列表
    l  (list)     ── 显示当前位置附近的 11 行源码
    ll (longlist) ── 显示当前函数完整源码
    w  (where)    ── 显示调用栈（从外到内），定位"我在哪里"
    a  (args)     ── 显示当前函数的所有参数值

  断点命令：
    b  <行号>     ── 在当前文件第 N 行设置断点，如 b 42
    b  <文件:行>  ── 跨文件设断点，如 b src/models/text_classifier.py:152
    cl <断点号>   ── 清除指定编号的断点（cl 1 清除断点1）
    tbreak <行号> ── 设置临时断点（触发一次后自动删除）

  执行 Python 表达式：
    !<表达式>     ── 在当前帧执行任意 Python 代码，如 !label = "tech"
                     （注意：会真实修改变量，可用来实验修复方案）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── 标准库 ──────────────────────────────────────────────────────────────────
import sys
import os

# 把项目根目录加入 sys.path，让 "from src.xxx import ..." 能正常工作
# 等价于在 IDE 里把根目录标记为 "Sources Root"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 被测模块 ─────────────────────────────────────────────────────────────────
import numpy as np
from src.models.text_classifier import TextClassifier

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【breakpoint() 是什么？】
#
# Python 3.7+ 内置函数，等价于旧写法 import pdb; pdb.set_trace()
# 执行到这一行时，程序暂停，控制台变成 (Pdb) 提示符。
#
# 可以通过环境变量切换调试后端：
#   PYTHONBREAKPOINT=0               ── 禁用所有断点（静默跳过）
#   PYTHONBREAKPOINT=ipdb.set_trace  ── 使用 ipdb（有语法高亮和 Tab 补全）
#
# 安装 ipdb：pip install ipdb
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def inspect_tfidf_step(classifier: TextClassifier, text: str) -> None:
    """演示场景：手动检查 TF-IDF 向量化的中间结果。

    真实调试场景还原：
        你发现某条"科技"文本被错误分类成了"财经"。
        怀疑是 TF-IDF 特征提取阶段出了问题。
        下面用 pdb 逐步检查 pipeline 的每个步骤。

    Args:
        classifier: 已训练好的 TextClassifier 实例。
        text: 待调试的输入文本。
    """
    print("\n" + "="*60)
    print("【调试场景】：检查 TF-IDF 向量化中间步骤")
    print(f"输入文本：{text!r}")
    print("="*60)

    # ── 从 pipeline 中取出各个步骤 ───────────────────────────────────────────
    # sklearn Pipeline 支持用名字访问步骤：pipeline.named_steps["步骤名"]
    # 对应 TextClassifier.__init__ 里定义的 ("tfidf", ...) 和 ("clf", ...)
    tfidf_vectorizer = classifier.pipeline.named_steps["tfidf"]
    nb_classifier   = classifier.pipeline.named_steps["clf"]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【第一个断点】：在向量化之前暂停
    #
    # 进入 (Pdb) 后，建议执行：
    #   p text                          ── 确认输入文本内容
    #   p tfidf_vectorizer.vocabulary_  ── 查看词汇表（字符 n-gram）
    #   p len(tfidf_vectorizer.vocabulary_)  ── 词汇表大小
    #   n                               ── 执行向量化
    #   p tfidf_matrix.shape            ── 看输出维度（应为 (1, vocab_size)）
    #   p tfidf_matrix.nnz              ── 非零元素数（命中了多少 n-gram）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[断点1] 即将执行 TF-IDF 向量化，输入 n 继续...")
    breakpoint()  # ← 程序在此暂停

    # TF-IDF 把文本转成稀疏矩阵（scipy sparse matrix）
    # transform() 只做向量化，不做训练；fit_transform() 才会更新词汇表
    tfidf_matrix = tfidf_vectorizer.transform([text])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【第二个断点】：检查概率分布
    #
    # 进入 (Pdb) 后，建议执行：
    #   p probabilities                          ── 看完整概率数组
    #   p list(zip(nb_classifier.classes_, probabilities[0]))  ── 类别+概率对照
    #   p probabilities[0].argmax()              ── 最高概率的索引
    #   p nb_classifier.classes_[probabilities[0].argmax()]    ── 预测类别
    #
    # 如果分类错误，可以进一步：
    #   !wrong_class_idx = list(nb_classifier.classes_).index("finance")
    #   p probabilities[0][wrong_class_idx]      ── 看"财经"的概率为何这么高
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[断点2] 即将执行朴素贝叶斯分类，输入 n 继续...")
    breakpoint()  # ← 程序在此暂停

    # predict_proba 接受 scipy 稀疏矩阵或 numpy 数组
    # 返回 shape (1, n_classes) 的 numpy 数组，每列对应一个类别的概率
    # 注意：MultinomialNB 的概率是经过 log 转换再 softmax 归一化的，不是原始频率
    probabilities: np.ndarray = nb_classifier.predict_proba(tfidf_matrix)

    # 整理结果：把类别名和概率配对，按概率降序排列
    class_probs = sorted(
        zip(nb_classifier.classes_, probabilities[0]),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n【分类结果】（概率从高到低）：")
    for rank, (cls, prob) in enumerate(class_probs, start=1):
        bar = "█" * int(prob * 40)  # 用方块字符绘制简单进度条
        print(f"  {rank}. {cls:<15} {prob:.4f}  {bar}")

    predicted = class_probs[0][0]
    print(f"\n最终预测：{predicted!r}")


def demonstrate_call_stack(classifier: TextClassifier, text: str) -> None:
    """演示 w (where) 命令：查看调用栈。

    调用栈告诉你"程序是如何执行到这里的"。
    层次越深（列表越长），说明嵌套越多。

    Args:
        classifier: 已训练好的 TextClassifier 实例。
        text: 待预测的文本。
    """
    print("\n" + "="*60)
    print("【调试场景】：用 w 命令查看调用栈")
    print("="*60)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【第三个断点】：在 predict() 被调用之前
    #
    # 进入 (Pdb) 后：
    #   w      ── 查看完整调用栈
    #          输出示例：
    #            -> debug_tools/pdb_debug_example.py(xxx)main()
    #            -> debug_tools/pdb_debug_example.py(yyy)demonstrate_call_stack()
    #            -> debug_tools/pdb_debug_example.py(zzz)<module>()  ← 当前帧
    #   s      ── 用 step 进入 classifier.predict() 内部
    #   w      ── 再次查看，调用栈会多一层 text_classifier.py
    #   l      ── 在 predict() 内部查看源码
    #   r      ── 跑完当前函数直接返回
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[断点3] 即将调用 classifier.predict()，输入 s 可进入函数内部...")
    breakpoint()  # ← 程序在此暂停

    result = classifier.predict(text)
    print(f"\npredict() 返回：{result}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【VS Code launch.json 配置示例】
#
# 把下面的内容保存到 .vscode/launch.json，
# 然后在编辑器里直接设置图形断点（点击行号左边），按 F5 启动调试。
#
# {
#   "version": "0.2.0",
#   "configurations": [
#     {
#       "name": "Python: pdb_debug_example",
#       "type": "debugpy",
#       "request": "launch",
#       "program": "${workspaceFolder}/debug_tools/pdb_debug_example.py",
#       "console": "integratedTerminal",
#       "cwd": "${workspaceFolder}",
#       "env": {
#         "PYTHONPATH": "${workspaceFolder}"
#       }
#     }
#   ]
# }
#
# VS Code 调试器（debugpy）和 pdb 的关系：
#   - debugpy 是 VS Code 内置的调试后端，遵循 DAP（Debug Adapter Protocol）
#   - 它会拦截 breakpoint()，用图形界面展示，而不是命令行 (Pdb) 提示符
#   - 功能更强：可以查看 Variables 面板、Watch 表达式、Call Stack 面板
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """主入口：依次运行三个调试场景。"""

    print("━" * 60)
    print("  pdb 断点调试教程  —  TextClassifier 调试实战")
    print("━" * 60)
    print()
    print("提示：")
    print("  • 跳过当前断点，继续到下一个：输入 c")
    print("  • 退出整个调试：输入 q")
    print("  • 不想进入交互调试（只看结果）：用 PYTHONBREAKPOINT=0 运行")
    print()

    # ── 初始化分类器（训练发生在 __init__ 里）───────────────────────────────
    print("正在初始化 TextClassifier（训练模型）...")
    classifier = TextClassifier()
    print("初始化完成。\n")

    # ── 场景一：检查被错误分类的"科技"文本 ────────────────────────────────
    # 这条文本提到了"股价"，可能导致分类器混淆科技和财经
    ambiguous_text = "人工智能芯片公司发布新品后股价大涨"
    inspect_tfidf_step(classifier, ambiguous_text)

    # ── 场景二：调用栈追踪 ────────────────────────────────────────────────
    tech_text = "量子计算机实现对传统算法的指数级加速"
    demonstrate_call_stack(classifier, tech_text)

    print("\n" + "="*60)
    print("教程结束。")
    print()
    print("【下一步学习建议】")
    print("  1. 用 PYTHONBREAKPOINT=ipdb.set_trace python debug_tools/pdb_debug_example.py")
    print("     体验 ipdb 的彩色输出和 Tab 补全")
    print("  2. 在 .vscode/launch.json 配置好后，尝试用 VS Code 图形调试器")
    print("  3. 阅读 debug_tools/memory_tracker.py，学习用 tracemalloc 追踪内存泄漏")
    print("="*60)


if __name__ == "__main__":
    main()
