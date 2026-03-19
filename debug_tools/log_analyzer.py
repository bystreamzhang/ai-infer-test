"""log_analyzer.py — 结构化日志分析工具（可运行教程）

═══════════════════════════════════════════════════════════════
  结构化日志 vs 传统文本日志
═══════════════════════════════════════════════════════════════

传统日志长这样：
    [2024-01-15 10:23:45] ERROR in app.py: Request failed, latency=1234ms

问题：这是给人看的，机器无法可靠解析。

结构化日志长这样（structlog 的 JSONRenderer 输出）：
    {"level": "error", "event": "request_failed", "latency_ms": 1234,
     "path": "/api/v1/classify", "timestamp": "2024-01-15T10:23:45.123Z"}

优势：
1. 每个字段有固定 key，可以直接用 json.loads() 解析
2. 可以按任意字段过滤、聚合（例如: WHERE path='/api/v1/classify'）
3. 天然适配 ELK Stack（Elasticsearch + Logstash + Kibana）
   ↳ Logstash 采集 → Elasticsearch 存储+索引 → Kibana 可视化

本工具用纯 Python 模拟 ELK 的核心分析功能。

═══════════════════════════════════════════════════════════════
  生产环境的日志分析流程
═══════════════════════════════════════════════════════════════

问题发生时的典型复盘流程（SRE/后端开发常用）：

1. 定位（What）: 错误率突增 → 筛 ERROR 日志 → 找到 event="request_failed"
2. 归因（Why）: 查慢请求 → latency_ms > 200ms 集中在某个端点
3. 修复（Fix）: 定位到代码，优化或回滚
4. 验证（Verify）: 部署后再次分析日志，确认错误率下降
5. 预防（Prevent）: 在 Grafana 设置告警规则（错误率 > 5% 时 PagerDuty 通知）

本工具覆盖了步骤 1-2 的分析能力。

运行方式：
    # 分析日志文件（见下方的测试命令）
    python debug_tools/log_analyzer.py --log-file reports/app.log

    # 只看最近 10 分钟内的日志
    python debug_tools/log_analyzer.py --log-file reports/app.log --time-window 10

    # 自动生成示例日志并分析（不需要先启动服务）
    python debug_tools/log_analyzer.py --demo
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ─── 数据类型定义 ──────────────────────────────────────────────────────────────
# 每条日志解析后的结构，用 TypedDict 而非 dataclass 是因为日志字段不固定
from typing import TypedDict


class LogEntry(TypedDict, total=False):
    """structlog JSON 日志的字段定义（total=False 表示所有字段可选）。

    structlog 的 JSONRenderer 会把每次 log.info("event", **kwargs) 的
    所有参数序列化成一个 JSON 对象。本项目的日志格式见 src/utils/logger.py。
    """
    level: str       # "info" / "warning" / "error" / "debug"
    event: str       # 事件名称，如 "request_received", "inference_complete"
    timestamp: str   # ISO 格式时间戳，如 "2024-01-15T10:23:45.123456Z"
    path: str        # HTTP 路径，如 "/api/v1/classify"
    latency_ms: float  # 请求延迟（毫秒）
    status_code: int   # HTTP 状态码
    logger: str        # 模块名，如 "src.app"
    filename: str      # 源文件名
    lineno: int        # 源码行号


# ─── 日志解析 ──────────────────────────────────────────────────────────────────

def parse_log_file(log_path: Path) -> list[LogEntry]:
    """逐行解析 structlog JSON 格式的日志文件。

    structlog 的 JSONRenderer 保证每行是一个完整的 JSON 对象，
    因此可以逐行 json.loads()，失败的行直接跳过（可能是启动信息等非 JSON 行）。

    Args:
        log_path: 日志文件路径。

    Returns:
        成功解析的 LogEntry 列表，顺序与文件行顺序一致。
    """
    entries: list[LogEntry] = []
    skipped = 0

    with log_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                # json.loads() 把 JSON 字符串解析成 Python dict
                entry: LogEntry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                # 非 JSON 行（如 uvicorn 启动时的纯文本日志）直接跳过
                skipped += 1

    if skipped > 0:
        print(f"  [提示] 跳过了 {skipped} 行非 JSON 格式的日志")

    return entries


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """将 ISO 格式时间戳字符串解析为 datetime 对象。

    structlog 的 TimeStamper(fmt="iso") 输出格式示例：
        "2024-01-15T10:23:45.123456Z"  （带微秒，UTC）
        "2024-01-15T10:23:45Z"         （无微秒，UTC）
        "2024-01-15T10:23:45.123456"   （无时区）

    Python 3.11+ 的 datetime.fromisoformat() 可以处理大部分 ISO 变体。

    Args:
        ts_str: ISO 格式时间戳字符串。

    Returns:
        解析成功返回 timezone-aware 的 datetime，失败返回 None。
    """
    if not ts_str:
        return None
    try:
        # Python 3.11 之前 fromisoformat 不支持末尾的 'Z'，需要替换
        normalized = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        # 如果没有时区信息，假设为 UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return None


# ─── 时间窗口过滤 ──────────────────────────────────────────────────────────────

def filter_by_time_window(
    entries: list[LogEntry],
    time_window_minutes: Optional[int],
) -> list[LogEntry]:
    """保留最近 N 分钟内的日志条目。

    生产环境中，值班工程师通常只关心"告警发生前后 30 分钟"的日志，
    time_window 参数就是这个"聚焦"机制。

    Args:
        entries: 待过滤的日志条目列表。
        time_window_minutes: 保留最近 N 分钟，None 表示不过滤。

    Returns:
        过滤后的日志条目列表。
    """
    if time_window_minutes is None:
        return entries

    # 找到日志中最新的时间戳作为"现在"（比用 datetime.now() 更合理——
    # 因为我们分析的可能是历史日志，wall clock time 不是参考点）
    latest_ts = None
    for entry in entries:
        ts = parse_timestamp(entry.get("timestamp", ""))
        if ts and (latest_ts is None or ts > latest_ts):
            latest_ts = ts

    if latest_ts is None:
        print("  [警告] 日志中没有可解析的时间戳，跳过时间过滤")
        return entries

    cutoff = latest_ts - timedelta(minutes=time_window_minutes)
    filtered = []
    for entry in entries:
        ts = parse_timestamp(entry.get("timestamp", ""))
        if ts is None or ts >= cutoff:
            filtered.append(entry)

    return filtered


# ─── 核心分析函数 ──────────────────────────────────────────────────────────────

def analyze_error_rate(entries: list[LogEntry]) -> dict:
    """统计错误率：ERROR 级别日志占总日志的比例。

    注意：structlog 把 level 存为小写字符串（"error"、"info"、"warning"）。

    Args:
        entries: 日志条目列表。

    Returns:
        包含 total, error_count, error_rate 的字典。
    """
    total = len(entries)
    if total == 0:
        return {"total": 0, "error_count": 0, "error_rate": 0.0}

    # Counter 是 dict 的子类，专门用来计数，这里按 level 分组
    level_counts = Counter(
        entry.get("level", "unknown").lower() for entry in entries
    )

    error_count = level_counts.get("error", 0)
    warning_count = level_counts.get("warning", 0)

    return {
        "total": total,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": level_counts.get("info", 0),
        "error_rate": round(error_count / total * 100, 2),
        "warning_rate": round(warning_count / total * 100, 2),
        "level_distribution": dict(level_counts),
    }


def analyze_slow_requests(
    entries: list[LogEntry],
    slow_threshold_ms: float = 200.0,
) -> dict:
    """统计慢请求：latency_ms 超过阈值的请求比例。

    200ms 是一个常见的 SLA（服务级别协议）基准——
    研究表明超过 200ms 的响应会让用户明显感觉到"卡顿"。

    Args:
        entries: 日志条目列表。
        slow_threshold_ms: 慢请求阈值（毫秒），默认 200ms。

    Returns:
        包含慢请求统计的字典。
    """
    # 只分析有 latency_ms 字段的条目（通常是请求完成日志）
    latency_entries = [
        e for e in entries if "latency_ms" in e and e["latency_ms"] is not None
    ]

    if not latency_entries:
        return {
            "total_with_latency": 0,
            "slow_count": 0,
            "slow_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }

    latencies = sorted(float(e["latency_ms"]) for e in latency_entries)
    slow_count = sum(1 for lat in latencies if lat > slow_threshold_ms)
    n = len(latencies)

    # 百分位数计算：p95 = 第 95% 位置的值
    # 注意：这是线性插值的简化版，生产中用 numpy.percentile 更精确
    def percentile(sorted_data: list[float], p: float) -> float:
        """计算百分位数（线性插值）。"""
        idx = (p / 100) * (len(sorted_data) - 1)
        lower, upper = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

    return {
        "total_with_latency": n,
        "slow_count": slow_count,
        "slow_rate": round(slow_count / n * 100, 2),
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "min_latency_ms": round(latencies[0], 2),
        "max_latency_ms": round(latencies[-1], 2),
        "p50_latency_ms": round(percentile(latencies, 50), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
        "p99_latency_ms": round(percentile(latencies, 99), 2),
        "threshold_ms": slow_threshold_ms,
    }


def analyze_endpoint_distribution(entries: list[LogEntry]) -> dict:
    """统计各端点的请求分布。

    通过 path 字段分组，统计每个 API 端点的调用次数和错误数。
    这能帮助识别"流量热点"和"高错误率端点"。

    Args:
        entries: 日志条目列表。

    Returns:
        按端点分组的统计字典。
    """
    # defaultdict(lambda: {"count": 0, "errors": 0, "latencies": []})
    # 比普通 dict 的好处：访问不存在的 key 时自动创建默认值，不用判断 key 是否存在
    endpoint_stats: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "errors": 0, "latencies": []}
    )

    for entry in entries:
        path = entry.get("path")
        if not path:
            continue  # 没有 path 字段的日志（如启动日志）跳过

        endpoint_stats[path]["count"] += 1

        if entry.get("level", "").lower() == "error":
            endpoint_stats[path]["errors"] += 1

        latency = entry.get("latency_ms")
        if latency is not None:
            endpoint_stats[path]["latencies"].append(float(latency))

    # 计算每个端点的平均延迟，然后清理掉原始数据（不需要返回原始列表）
    result = {}
    for path, stats in endpoint_stats.items():
        latencies = stats["latencies"]
        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
        result[path] = {
            "request_count": stats["count"],
            "error_count": stats["errors"],
            "error_rate": round(stats["errors"] / stats["count"] * 100, 2),
            "avg_latency_ms": avg_lat,
        }

    # 按请求量降序排列
    return dict(sorted(result.items(), key=lambda x: x[1]["request_count"], reverse=True))


def analyze_requests_per_minute(entries: list[LogEntry]) -> dict:
    """统计每分钟请求数时间序列。

    时间序列分析是识别"流量突刺"的关键——
    告警通常在某分钟的 QPS（每秒请求数）超过阈值时触发。

    时间桶（Time Bucketing）原理：
    - 将每个时间戳截断到分钟精度（去掉秒和微秒）
    - 用 Counter 统计每个分钟桶中的日志数量
    - 结果就是 QPS（这里是 QPM = Queries Per Minute）

    Args:
        entries: 日志条目列表。

    Returns:
        包含时间序列数据的字典。
    """
    minute_counts: Counter = Counter()

    for entry in entries:
        ts = parse_timestamp(entry.get("timestamp", ""))
        if ts is None:
            continue

        # 截断到分钟：replace(second=0, microsecond=0)
        # 例如 "10:23:45.123" → "10:23:00"
        minute_bucket = ts.replace(second=0, microsecond=0)
        # isoformat() 转成字符串作为 key，方便后续排序和展示
        minute_counts[minute_bucket.isoformat()] += 1

    if not minute_counts:
        return {"time_series": [], "peak_minute": None, "peak_count": 0}

    # 按时间排序
    sorted_series = sorted(minute_counts.items())
    peak_minute, peak_count = max(minute_counts.items(), key=lambda x: x[1])

    return {
        "time_series": [{"minute": m, "count": c} for m, c in sorted_series],
        "peak_minute": peak_minute,
        "peak_count": peak_count,
        "total_minutes": len(sorted_series),
    }


def collect_recent_errors(entries: list[LogEntry], max_errors: int = 5) -> list[dict]:
    """收集最近的 ERROR 日志详情，用于问题报告中的"错误样本"。

    Args:
        entries: 日志条目列表。
        max_errors: 最多返回多少条错误样本。

    Returns:
        错误日志列表（最多 max_errors 条，时间倒序）。
    """
    errors = [e for e in entries if e.get("level", "").lower() == "error"]
    # 取最后 N 条（最近发生的错误）
    recent = errors[-max_errors:]
    recent.reverse()  # 最新的排最前面

    result = []
    for e in recent:
        result.append({
            "timestamp": e.get("timestamp", "unknown"),
            "event": e.get("event", "unknown"),
            "path": e.get("path", "N/A"),
            "message": str(e.get("message", e.get("event", ""))),
            "location": f"{e.get('filename', '?')}:{e.get('lineno', '?')}",
        })
    return result


# ─── 报告格式化 ────────────────────────────────────────────────────────────────

def print_report(
    error_stats: dict,
    latency_stats: dict,
    endpoint_stats: dict,
    rpm_stats: dict,
    recent_errors: list[dict],
    log_path: str,
    time_window: Optional[int],
) -> None:
    """将分析结果格式化输出为问题摘要报告。

    Args:
        error_stats: analyze_error_rate() 的返回值。
        latency_stats: analyze_slow_requests() 的返回值。
        endpoint_stats: analyze_endpoint_distribution() 的返回值。
        rpm_stats: analyze_requests_per_minute() 的返回值。
        recent_errors: collect_recent_errors() 的返回值。
        log_path: 被分析的日志文件路径（仅用于展示）。
        time_window: 时间窗口（分钟），None 表示全量分析。
    """
    SEP = "═" * 60
    sep = "─" * 60

    print(f"\n{SEP}")
    print("  日志分析报告 (Log Analysis Report)")
    print(SEP)

    # 文件信息
    window_str = f"最近 {time_window} 分钟" if time_window else "全量"
    print(f"  分析文件  : {log_path}")
    print(f"  时间范围  : {window_str}")
    print(f"  日志总条数: {error_stats['total']}")

    # ── 1. 错误率 ──
    print(f"\n{'[ 1. 错误率分析 ]':}")
    print(sep)
    total = error_stats["total"]
    if total == 0:
        print("  无日志数据")
    else:
        err_rate = error_stats["error_rate"]
        warn_rate = error_stats["warning_rate"]

        # 用 emoji 指示严重程度（视觉化的告警分级）
        if err_rate >= 10:
            status = "CRITICAL"
            indicator = "[!!]"
        elif err_rate >= 5:
            status = "WARNING"
            indicator = "[!] "
        else:
            status = "OK"
            indicator = "[OK]"

        print(f"  {indicator} 错误率: {err_rate}%  ({error_stats['error_count']}/{total})")
        print(f"       告警率: {warn_rate}%  ({error_stats['warning_count']}/{total})")
        print(f"       状态  : {status}")
        print(f"  日志级别分布: {error_stats['level_distribution']}")

    # ── 2. 延迟分析 ──
    print(f"\n[ 2. 延迟分析 (慢请求阈值: {latency_stats.get('threshold_ms', 200)}ms) ]")
    print(sep)
    n_lat = latency_stats["total_with_latency"]
    if n_lat == 0:
        print("  无延迟数据（日志中没有 latency_ms 字段）")
    else:
        slow_rate = latency_stats["slow_rate"]
        if slow_rate >= 20:
            lat_indicator = "[!!]"
        elif slow_rate >= 5:
            lat_indicator = "[!] "
        else:
            lat_indicator = "[OK]"

        print(f"  {lat_indicator} 慢请求率: {slow_rate}%  ({latency_stats['slow_count']}/{n_lat})")
        print(f"       平均延迟: {latency_stats['avg_latency_ms']}ms")
        print(f"       P50 延迟: {latency_stats['p50_latency_ms']}ms")
        print(f"       P95 延迟: {latency_stats['p95_latency_ms']}ms")
        print(f"       P99 延迟: {latency_stats['p99_latency_ms']}ms")
        print(f"       最大延迟: {latency_stats['max_latency_ms']}ms")

    # ── 3. 端点分布 ──
    print(f"\n[ 3. 各端点请求分布 ]")
    print(sep)
    if not endpoint_stats:
        print("  无端点数据（日志中没有 path 字段）")
    else:
        print(f"  {'端点':<35} {'请求数':>6} {'错误数':>6} {'错误率':>7} {'均延迟':>9}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*7} {'-'*9}")
        for path, stats in endpoint_stats.items():
            avg_lat = f"{stats['avg_latency_ms']}ms" if stats['avg_latency_ms'] else "N/A"
            err_flag = " <--" if stats["error_rate"] >= 10 else ""
            print(
                f"  {path:<35} {stats['request_count']:>6} "
                f"{stats['error_count']:>6} {stats['error_rate']:>6.1f}% "
                f"{avg_lat:>9}{err_flag}"
            )

    # ── 4. 每分钟请求数 ──
    print(f"\n[ 4. 每分钟请求数 (QPM 时间序列) ]")
    print(sep)
    if not rpm_stats["time_series"]:
        print("  无时间序列数据")
    else:
        print(f"  峰值: {rpm_stats['peak_count']} 次/分 @ {rpm_stats['peak_minute']}")
        print(f"  分钟数: {rpm_stats['total_minutes']}")
        print()

        # 用 ASCII 柱状图可视化时间序列
        # 这是一个"穷人版 Grafana"——在终端里看趋势
        peak = rpm_stats["peak_count"]
        bar_width = 30  # 柱状图最大宽度（字符数）
        print(f"  {'时间':<25} {'请求数':>5}  分布图")
        print(f"  {'-'*25} {'-'*5}  {'-'*bar_width}")
        for item in rpm_stats["time_series"]:
            count = item["count"]
            bar_len = int(count / peak * bar_width) if peak > 0 else 0
            bar = "█" * bar_len
            # 截断时间戳显示（只显示时间部分，不显示日期）
            time_str = item["minute"]
            if "T" in time_str:
                time_str = time_str.split("T")[1][:8]  # "HH:MM:SS" 部分
            print(f"  {time_str:<25} {count:>5}  {bar}")

    # ── 5. 最近错误样本 ──
    print(f"\n[ 5. 最近错误样本 (最多 {len(recent_errors)} 条) ]")
    print(sep)
    if not recent_errors:
        print("  无错误日志，服务运行正常!")
    else:
        for i, err in enumerate(recent_errors, 1):
            print(f"  [{i}] {err['timestamp']}")
            print(f"      事件  : {err['event']}")
            print(f"      路径  : {err['path']}")
            print(f"      位置  : {err['location']}")
            print()

    # ── 总结与建议 ──
    print(f"[ 总结与建议 ]")
    print(sep)

    issues = []
    if error_stats["total"] > 0 and error_stats["error_rate"] >= 5:
        issues.append(f"错误率偏高 ({error_stats['error_rate']}%) → 检查 ERROR 日志中的 exception 字段")
    if latency_stats["total_with_latency"] > 0 and latency_stats["slow_rate"] >= 20:
        issues.append(f"慢请求过多 ({latency_stats['slow_rate']}%) → 运行 cprofile_analysis.py 定位瓶颈")
    if latency_stats["p99_latency_ms"] > 1000:
        issues.append(f"P99 延迟过高 ({latency_stats['p99_latency_ms']}ms) → 检查是否有超时或资源竞争")

    # 检查是否有高错误率端点
    for path, stats in endpoint_stats.items():
        if stats["error_rate"] >= 10 and stats["request_count"] >= 5:
            issues.append(f"端点 {path} 错误率 {stats['error_rate']}% → 重点排查该路由的异常处理")

    if issues:
        print("  发现以下问题：")
        for issue in issues:
            print(f"  !! {issue}")
    else:
        print("  未发现明显异常，服务运行健康。")

    print(SEP)


# ─── 示例日志生成（Demo 模式） ─────────────────────────────────────────────────

def generate_demo_log(output_path: Path, n_entries: int = 200) -> None:
    """生成一份示例 JSON 日志文件，用于演示分析功能。

    当没有真实日志时，用这个函数创造数据。
    模拟了一个有性能问题（慢请求）和偶发错误的服务场景。

    Args:
        output_path: 日志文件写入路径。
        n_entries: 生成的日志条目数。
    """
    print(f"[Demo] 正在生成 {n_entries} 条示例日志到 {output_path} ...")

    # 模拟从某个时间点开始，每 0.5-3 秒一条日志
    base_time = datetime.now(timezone.utc) - timedelta(minutes=30)

    paths = [
        "/api/v1/classify",
        "/api/v1/batch/classify",
        "/api/v1/generate",
        "/api/v1/health",
        "/api/v1/models",
        "/api/v1/metrics",
    ]

    # 各端点的基础延迟（模拟不同端点的性能特征）
    path_base_latency = {
        "/api/v1/classify": 50,
        "/api/v1/batch/classify": 180,  # 批量分类本来就慢
        "/api/v1/generate": 120,
        "/api/v1/health": 5,
        "/api/v1/models": 3,
        "/api/v1/metrics": 8,
    }

    entries = []
    current_time = base_time

    for i in range(n_entries):
        current_time += timedelta(seconds=random.uniform(0.3, 2.0))
        path = random.choices(
            paths,
            weights=[40, 15, 20, 10, 5, 10],  # 分类请求最多
        )[0]

        base_lat = path_base_latency[path]
        # 添加随机噪声，偶尔产生慢请求（模拟 GC 暂停、锁竞争等）
        if random.random() < 0.1:  # 10% 概率慢请求
            latency = base_lat + random.uniform(200, 800)
        else:
            latency = base_lat + random.gauss(0, base_lat * 0.2)
            latency = max(1.0, latency)

        # 错误率约 5%（集中在批量分类和文本生成端点）
        is_error = (
            random.random() < 0.12
            if path in ("/api/v1/batch/classify", "/api/v1/generate")
            else random.random() < 0.02
        )

        level = "error" if is_error else "info"
        event = "inference_failed" if is_error else "inference_complete"

        entry = {
            "level": level,
            "event": event,
            "timestamp": current_time.isoformat(),
            "path": path,
            "latency_ms": round(latency, 2),
            "status_code": 500 if is_error else 200,
            "logger": "src.app",
            "filename": "app.py",
            "lineno": 280 if is_error else 290,
        }

        if is_error:
            entry["message"] = random.choice([
                "Model inference timeout",
                "Input validation failed",
                "Concurrency limit exceeded",
                "Memory allocation error",
            ])

        entries.append(entry)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[Demo] 日志已生成，共 {len(entries)} 条")


# ─── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """命令行入口函数。

    支持两种模式：
    1. 分析模式：--log-file 指定日志文件
    2. Demo 模式：--demo 自动生成示例日志并分析
    """
    # argparse 是 Python 标准库的命令行参数解析器
    # 类比：它就是 C++ 里的 getopt，但更 Pythonic
    parser = argparse.ArgumentParser(
        description="结构化日志分析工具 — 分析 structlog JSON 格式日志",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python debug_tools/log_analyzer.py --demo
  python debug_tools/log_analyzer.py --log-file reports/app.log
  python debug_tools/log_analyzer.py --log-file reports/app.log --time-window 10
        """,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="日志文件路径（structlog JSON 格式，每行一个 JSON 对象）",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=None,
        metavar="MINUTES",
        help="只分析最近 N 分钟的日志（默认: 全量分析）",
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=200.0,
        metavar="MS",
        help="慢请求阈值（毫秒，默认: 200）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo 模式：自动生成示例日志并分析（无需真实日志文件）",
    )

    args = parser.parse_args()

    # ── Demo 模式 ──────────────────────────────────────────────────────────────
    if args.demo:
        demo_log_path = Path("reports/demo_app.log")
        generate_demo_log(demo_log_path, n_entries=300)
        log_path = demo_log_path
    elif args.log_file:
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"错误: 日志文件不存在: {log_path}", file=sys.stderr)
            print("提示: 运行 --demo 模式会自动生成示例日志", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        print("\n错误: 请指定 --log-file 或使用 --demo 模式", file=sys.stderr)
        sys.exit(1)

    # ── 解析日志 ──────────────────────────────────────────────────────────────
    print(f"\n正在解析日志文件: {log_path}")
    entries = parse_log_file(log_path)
    print(f"解析完成，共 {len(entries)} 条日志")

    # ── 时间过滤 ──────────────────────────────────────────────────────────────
    if args.time_window:
        print(f"按时间窗口过滤: 最近 {args.time_window} 分钟")
        entries = filter_by_time_window(entries, args.time_window)
        print(f"过滤后剩余 {len(entries)} 条日志")

    if not entries:
        print("过滤后没有日志数据，请检查时间窗口设置")
        sys.exit(0)

    # ── 执行分析 ──────────────────────────────────────────────────────────────
    print("正在分析...")
    error_stats = analyze_error_rate(entries)
    latency_stats = analyze_slow_requests(entries, slow_threshold_ms=args.slow_threshold)
    endpoint_stats = analyze_endpoint_distribution(entries)
    rpm_stats = analyze_requests_per_minute(entries)
    recent_errors = collect_recent_errors(entries, max_errors=5)

    # ── 输出报告 ──────────────────────────────────────────────────────────────
    print_report(
        error_stats=error_stats,
        latency_stats=latency_stats,
        endpoint_stats=endpoint_stats,
        rpm_stats=rpm_stats,
        recent_errors=recent_errors,
        log_path=str(log_path),
        time_window=args.time_window,
    )


if __name__ == "__main__":
    # 这个条件语句的意义：
    # - 当 python log_analyzer.py 直接运行时，__name__ == "__main__"，执行 main()
    # - 当被 import 时（如测试中 from log_analyzer import parse_log_file），
    #   __name__ == "debug_tools.log_analyzer"，不自动执行 main()
    # 这是 Python 模块的标准组织方式
    main()
