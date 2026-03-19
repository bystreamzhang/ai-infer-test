"""火焰图生成器 — 使用 py-spy 对运行中的服务进行采样式 profiling。

本文件是一个"启动器 + 说明文档"：
  1. 用 subprocess 启动被测服务（uvicorn）
  2. 用 httpx 向服务发送负载（让 CPU 有东西可采样）
  3. 同时打印 py-spy 命令，供你在另一个终端手动执行
  4. 等待一段时间后自动关闭服务

运行方式：
    python debug_tools/flame_graph_gen.py

注意：py-spy 需要单独在另一个终端运行，本脚本会告诉你怎么做。
"""

# ─────────────────────────────────────────────────────────────────────────────
# 【概念一】什么是"采样式 profiler"？
#
# profiler 有两种主流实现方式：
#
#   插桩式（Instrumentation）：在每个函数的入口/出口插入钩子，精确记录每次调用。
#     代表：Python 内置的 cProfile
#     优点：精确，能统计调用次数
#     缺点：本身有开销（最多让程序慢 2-3x），影响测量结果
#
#   采样式（Sampling）：每隔固定时间（如 10ms）"拍一张快照"，记录当前调用栈。
#     代表：py-spy、Java 的 async-profiler、Linux 的 perf
#     优点：开销极小（通常 <1%），不影响生产环境
#     缺点：统计性质，低频函数可能被漏掉
#
# 类比：
#   插桩式 = 给每个员工装门禁，精确记录每次进出
#   采样式 = 每隔10分钟拍一张办公室全景，看谁在什么位置
#
# py-spy 是目前 Python 生态最流行的采样式 profiler，特点：
#   - 无需修改被测代码
#   - 可以 attach 到已运行的进程（非侵入式）
#   - 直接生成 SVG 格式的交互式火焰图
#   - 安装：pip install py-spy
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 【概念二】如何读火焰图？
#
#   火焰图的 x 轴：采样栈的聚合宽度
#     → 一个函数的矩形越宽，表示它（及其子调用）占 CPU 时间越多
#     → x 轴没有时间顺序！宽度 = 采样次数 / 总采样次数
#
#   火焰图的 y 轴：调用深度（调用栈层数）
#     → 最底层是程序入口（如 main / uvicorn 事件循环）
#     → 越往上是越深的子调用
#     → 栈顶（最上面）的函数是实际执行 CPU 指令的地方
#
#   如何找瓶颈？
#     1. 找"平顶山"：栈顶宽但上方没有子调用的函数 = 自身 CPU 消耗大
#     2. 找"宽柱子"：某个中间层函数宽度大 = 它的某个子调用很耗时
#     3. 特别关注宽度 > 5% 的函数
#
#   交互技巧（在浏览器打开 SVG 时）：
#     - 点击某个矩形 → 以该函数为根重新缩放，方便聚焦
#     - 搜索框输入函数名 → 高亮所有匹配
#     - Ctrl+F → 浏览器搜索
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 【概念三】py-spy 的两种主要模式
#
#   py-spy record（生成火焰图）：
#     py-spy record -o output.svg --pid <PID> --duration 30
#     → 采样 30 秒，生成 output.svg
#     → --rate 100 表示每秒采样100次（默认100）
#     → --native 同时采样 C 扩展（如 numpy/sklearn 内部），更全面
#
#   py-spy top（实时 top 视图）：
#     py-spy top --pid <PID>
#     → 类似 Linux 的 top 命令，实时显示各函数的 CPU 占比
#     → 适合快速定位"现在谁最耗 CPU"，不需要生成文件
#
#   两者对比：
#     top   = 实时诊断，适合"现在服务慢，快速看看哪里"
#     record = 详细分析，适合"生成报告，深入分析调用链"
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path

# httpx 是本项目用于测试的异步 HTTP 客户端，这里用同步接口发负载
try:
    import httpx
except ImportError:
    print("❌ 缺少 httpx：pip install httpx")
    sys.exit(1)

# ─── 常量配置 ──────────────────────────────────────────────────────────────────

SERVICE_HOST = "http://127.0.0.1:8000"
SERVICE_STARTUP_WAIT = 3.0       # 等待 uvicorn 启动的秒数
LOAD_DURATION = 30               # 制造负载的持续时间（秒）
LOAD_REQUESTS_PER_SECOND = 10    # 每秒发送的请求数
REPORTS_DIR = Path("reports")

# py-spy 采样参数
PYSPY_DURATION = 25       # 采样时长（秒），比负载略短，确保采到有效数据
PYSPY_RATE = 100          # 每秒采样次数（100 是默认值，已足够）

# 测试用的文本样本，覆盖不同类别，让 classifier 有东西处理
TEST_TEXTS = [
    "machine learning algorithms for classification",
    "python programming tips and tricks",
    "deep neural network architecture design",
    "web scraping with beautiful soup",
    "statistical analysis of experimental data",
    "database query optimization techniques",
    "natural language processing with transformers",
    "software design patterns in practice",
    "computer vision object detection models",
    "cloud infrastructure deployment automation",
]


def ensure_reports_dir() -> None:
    """确保 reports/ 目录存在。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def start_service() -> subprocess.Popen:
    """用 subprocess 启动 uvicorn 服务。

    Returns:
        启动的子进程对象，稍后用于获取 PID 和终止进程。

    Note:
        stdout/stderr 重定向到 DEVNULL，避免干扰终端输出。
        实际生产中你可能想重定向到日志文件。
    """
    print("🚀 正在启动 uvicorn 服务...")

    # subprocess.Popen 非阻塞地启动一个子进程
    # sys.executable 是当前 Python 解释器的路径，确保用同一个虚拟环境
    # -m uvicorn 等价于 python -m uvicorn（通过模块名运行）
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.app:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--workers", "1",  # 单 worker，让 py-spy 只需要 attach 一个进程
        ],
        stdout=subprocess.DEVNULL,  # 丢弃标准输出，避免干扰
        stderr=subprocess.DEVNULL,  # 丢弃错误输出
        # 在 Windows 上，creationflags 让子进程独立于当前终端
        # 在 Linux/Mac 上这个参数不存在，需要条件判断
        **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}),
    )

    print(f"   子进程 PID: {proc.pid}")
    print(f"   等待 {SERVICE_STARTUP_WAIT}s 让服务完成初始化（模型加载）...")
    time.sleep(SERVICE_STARTUP_WAIT)

    # 验证服务是否正常启动
    try:
        resp = httpx.get(f"{SERVICE_HOST}/api/v1/health", timeout=5.0)
        if resp.status_code == 200:
            print(f"   ✅ 服务已就绪，健康检查通过：{resp.json()}")
        else:
            print(f"   ⚠️  服务响应异常，状态码：{resp.status_code}")
    except httpx.ConnectError:
        print("   ❌ 无法连接服务，可能启动失败。请检查 uvicorn 是否正确安装。")
        proc.terminate()
        sys.exit(1)

    return proc


def print_pyspy_instructions(pid: int) -> None:
    """打印 py-spy 使用说明，引导用户在另一个终端执行。

    Args:
        pid: 被测服务的进程 ID。
    """
    output_svg = REPORTS_DIR / "flamegraph.svg"

    print("\n" + "=" * 70)
    print("🔥 py-spy 火焰图采样命令")
    print("=" * 70)
    print()
    print("请【立即】在另一个终端中运行以下命令：")
    print()

    # ── 方式一：生成火焰图 SVG ──────────────────────────────────────────────
    print("【方式一】生成交互式火焰图（推荐）：")
    print(f"  py-spy record \\")
    print(f"    -o {output_svg} \\")
    print(f"    --pid {pid} \\")
    print(f"    --duration {PYSPY_DURATION} \\")
    print(f"    --rate {PYSPY_RATE} \\")
    print(f"    --native")
    print()
    print(f"  完成后用浏览器打开：{output_svg.resolve()}")
    print()

    # ── 方式二：实时 top 视图 ───────────────────────────────────────────────
    print("【方式二】实时查看热点函数（快速诊断）：")
    print(f"  py-spy top --pid {pid}")
    print()

    # ── Windows 特殊说明 ────────────────────────────────────────────────────
    if sys.platform == "win32":
        print("⚠️  Windows 提示：")
        print("  py-spy 在 Windows 上需要管理员权限（以管理员身份运行终端）")
        print("  或者使用 --nonblocking 参数降低权限要求：")
        print(f"  py-spy record -o {output_svg} --pid {pid} --nonblocking --duration {PYSPY_DURATION}")
        print()

    print("=" * 70)
    print(f"⏱  本脚本将在 {LOAD_DURATION} 秒内持续发送负载，请在此期间运行上述命令")
    print("=" * 70)
    print()


def send_load(stop_event: threading.Event) -> None:
    """在后台线程中持续向服务发送 HTTP 请求，制造 CPU 负载。

    Args:
        stop_event: 收到停止信号时设置，线程据此退出循环。

    Note:
        使用同步 httpx 客户端（而非 asyncio），因为这里在普通线程中运行。
        每次请求之间 sleep 一小段时间，控制 QPS（每秒请求数）。
    """
    # 用 httpx.Client 作为 context manager，自动管理连接池
    with httpx.Client(base_url=SERVICE_HOST, timeout=10.0) as client:
        request_count = 0
        error_count = 0
        interval = 1.0 / LOAD_REQUESTS_PER_SECOND  # 每次请求的间隔时间（秒）

        while not stop_event.is_set():
            # 轮流使用不同的文本，避免缓存层把所有请求都缓存掉
            # （缓存命中时 CPU 消耗很低，采样不到真实的推理路径）
            text = TEST_TEXTS[request_count % len(TEST_TEXTS)]

            try:
                resp = client.post(
                    "/api/v1/classify",
                    json={"text": text},
                )
                if resp.status_code == 200:
                    request_count += 1
                else:
                    error_count += 1
            except (httpx.ConnectError, httpx.TimeoutException):
                error_count += 1

            # 每 50 次请求打印一次进度
            if request_count > 0 and request_count % 50 == 0:
                print(f"   📊 已发送 {request_count} 次请求，错误 {error_count} 次")

            # 控制请求速率：sleep 让出 GIL，避免把整个 Python 进程打满
            time.sleep(interval)

    print(f"   负载线程结束：共发送 {request_count} 次请求，错误 {error_count} 次")


def stop_service(proc: subprocess.Popen) -> None:
    """安全终止 uvicorn 子进程。

    Args:
        proc: start_service() 返回的子进程对象。

    Note:
        先发 SIGTERM（优雅停机），等 3 秒后若还在运行则 SIGKILL（强制终止）。
        在 Windows 上没有 SIGTERM，直接用 terminate()。
    """
    print("\n🛑 正在停止服务...")

    if sys.platform == "win32":
        # Windows：直接终止（taskkill 方式）
        proc.terminate()
    else:
        # Unix/Mac：先 SIGTERM，再等待
        proc.send_signal(signal.SIGTERM)

    try:
        proc.wait(timeout=5)
        print("   ✅ 服务已正常停止")
    except subprocess.TimeoutExpired:
        print("   ⚠️  服务未在 5 秒内停止，强制终止...")
        proc.kill()
        proc.wait()
        print("   ✅ 服务已强制停止")


def print_summary(pid: int) -> None:
    """打印整体流程回顾和后续步骤建议。

    Args:
        pid: 被测服务的进程 ID（用于说明文档）。
    """
    output_svg = REPORTS_DIR / "flamegraph.svg"

    print("\n" + "=" * 70)
    print("📋 流程回顾")
    print("=" * 70)
    print("""
本脚本做了以下事情：
  1. 用 subprocess 启动 uvicorn（PID: {pid}）
  2. 打印了 py-spy 命令，等待你手动执行
  3. 用 httpx 以 {qps} QPS 持续发送分类请求 {dur} 秒
  4. 优雅停止 uvicorn

如果你已经执行了 py-spy record 命令，现在可以：
  • 用浏览器打开 {svg} 查看火焰图
  • 在火焰图中寻找宽而平的"平顶山"（自身 CPU 消耗大）
  • 关注 tfidf、naive_bayes、predict_proba 等 sklearn 内部函数的宽度比例

如果没来得及执行，可以手动重新运行本脚本，或者：
  • 手动启动服务：uvicorn src.app:app --host 127.0.0.1 --port 8000
  • 然后直接运行 py-spy 命令（用上面打印的 PID 换成新的进程 PID）

进一步学习：
  • py-spy 文档：https://github.com/benfred/py-spy
  • 火焰图原理（Brendan Gregg）：https://www.brendangregg.com/flamegraphs.html
  • 与 cProfile 对比：debug_tools/cprofile_analysis.py
""".format(
        pid=pid,
        qps=LOAD_REQUESTS_PER_SECOND,
        dur=LOAD_DURATION,
        svg=output_svg.resolve(),
    ))


def main() -> None:
    """主流程：启动服务 → 打印指令 → 发送负载 → 停止服务 → 打印总结。"""
    ensure_reports_dir()

    # ── Step 1：启动服务 ────────────────────────────────────────────────────
    proc = start_service()
    pid = proc.pid

    # ── Step 2：打印 py-spy 命令 ────────────────────────────────────────────
    # 在发负载之前打印，给用户时间去另一个终端执行命令
    print_pyspy_instructions(pid)

    # ── Step 3：在后台线程发负载 ────────────────────────────────────────────
    # 使用 threading.Event 作为停止信号，比直接用全局变量更清晰
    stop_event = threading.Event()

    # daemon=True：主线程退出时，后台线程自动终止（不需要手动 join）
    load_thread = threading.Thread(
        target=send_load,
        args=(stop_event,),
        daemon=True,
        name="load-generator",
    )

    print(f"⚡ 开始发送负载，持续 {LOAD_DURATION} 秒...")
    print(f"   请现在去另一个终端执行 py-spy 命令！")
    print()

    load_thread.start()

    # ── Step 4：等待负载时间结束 ────────────────────────────────────────────
    # 每秒打印一次倒计时，让用户知道还有多少时间
    for remaining in range(LOAD_DURATION, 0, -5):
        if remaining % 10 == 0 or remaining <= 5:
            print(f"   ⏳ 还剩 {remaining} 秒...")
        time.sleep(min(5, remaining))

    # 发出停止信号，等待负载线程退出
    stop_event.set()
    load_thread.join(timeout=3)

    # ── Step 5：停止服务 ────────────────────────────────────────────────────
    stop_service(proc)

    # ── Step 6：打印总结 ────────────────────────────────────────────────────
    print_summary(pid)


if __name__ == "__main__":
    main()
