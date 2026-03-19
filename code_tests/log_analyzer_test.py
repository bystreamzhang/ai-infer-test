"""log_analyzer.py 功能验证测试

直接运行此文件检查各核心函数是否工作正常。
不依赖 pytest，可以 python code_tests/log_analyzer_test.py 直接运行。
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 将项目根目录加入路径，使 import 正常工作
sys.path.insert(0, str(Path(__file__).parent.parent))

from debug_tools.log_analyzer import (
    analyze_endpoint_distribution,
    analyze_error_rate,
    analyze_requests_per_minute,
    analyze_slow_requests,
    collect_recent_errors,
    filter_by_time_window,
    parse_log_file,
    parse_timestamp,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


def make_entries(specs: list[dict]) -> list[dict]:
    """生成测试用的日志条目列表。"""
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    result = []
    for i, spec in enumerate(specs):
        entry = {
            "level": spec.get("level", "info"),
            "event": spec.get("event", "request"),
            "timestamp": (base + timedelta(seconds=i * 10)).isoformat(),
        }
        if "path" in spec:
            entry["path"] = spec["path"]
        if "latency_ms" in spec:
            entry["latency_ms"] = spec["latency_ms"]
        result.append(entry)
    return result


def test_parse_timestamp():
    print("\n--- test_parse_timestamp ---")

    cases = [
        ("2024-01-15T10:23:45.123456Z", True),
        ("2024-01-15T10:23:45Z", True),
        ("2024-01-15T10:23:45", True),
        ("not-a-timestamp", False),
        ("", False),
    ]
    for ts_str, expect_ok in cases:
        result = parse_timestamp(ts_str)
        ok = (result is not None) == expect_ok
        print(f"  {PASS if ok else FAIL} parse_timestamp({ts_str!r}) → {result}")
        if not ok:
            print(f"    期望: {'成功' if expect_ok else '失败'}")


def test_parse_log_file():
    print("\n--- test_parse_log_file ---")

    lines = [
        '{"level": "info", "event": "startup"}',
        '{"level": "error", "event": "failure"}',
        "这不是JSON",
        "",
        '{"level": "info", "event": "ok"}',
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))
        tmp_path = Path(f.name)

    entries = parse_log_file(tmp_path)
    tmp_path.unlink()  # 清理临时文件

    ok = len(entries) == 3
    print(f"  {PASS if ok else FAIL} 解析 5 行（1 非JSON + 1 空行）→ 期望 3 条，实际 {len(entries)} 条")

    ok2 = entries[1]["level"] == "error"
    print(f"  {PASS if ok2 else FAIL} 第 2 条日志 level == 'error'")


def test_analyze_error_rate():
    print("\n--- test_analyze_error_rate ---")

    # 10 条：8 info + 2 error → 错误率 20%
    entries = make_entries(
        [{"level": "info"}] * 8 + [{"level": "error"}] * 2
    )
    stats = analyze_error_rate(entries)

    ok1 = stats["total"] == 10
    ok2 = stats["error_count"] == 2
    ok3 = abs(stats["error_rate"] - 20.0) < 0.1

    print(f"  {PASS if ok1 else FAIL} total == 10，实际 {stats['total']}")
    print(f"  {PASS if ok2 else FAIL} error_count == 2，实际 {stats['error_count']}")
    print(f"  {PASS if ok3 else FAIL} error_rate == 20.0%，实际 {stats['error_rate']}%")

    # 边界：空列表
    empty_stats = analyze_error_rate([])
    ok4 = empty_stats["total"] == 0 and empty_stats["error_rate"] == 0.0
    print(f"  {PASS if ok4 else FAIL} 空列表 → error_rate == 0.0")


def test_analyze_slow_requests():
    print("\n--- test_analyze_slow_requests ---")

    # 4 个正常请求 + 1 个慢请求（250ms）
    entries = make_entries([
        {"latency_ms": 50},
        {"latency_ms": 100},
        {"latency_ms": 80},
        {"latency_ms": 120},
        {"latency_ms": 250},  # 慢请求
    ])
    stats = analyze_slow_requests(entries, slow_threshold_ms=200.0)

    ok1 = stats["slow_count"] == 1
    ok2 = abs(stats["slow_rate"] - 20.0) < 0.1
    ok3 = stats["p99_latency_ms"] >= 200  # P99 应该接近 250

    print(f"  {PASS if ok1 else FAIL} slow_count == 1，实际 {stats['slow_count']}")
    print(f"  {PASS if ok2 else FAIL} slow_rate == 20.0%，实际 {stats['slow_rate']}%")
    print(f"  {PASS if ok3 else FAIL} p99 >= 200ms，实际 {stats['p99_latency_ms']}ms")

    # 无 latency_ms 字段的条目应被跳过
    no_lat_entries = make_entries([{"level": "info"}] * 5)
    no_lat_stats = analyze_slow_requests(no_lat_entries)
    ok4 = no_lat_stats["total_with_latency"] == 0
    print(f"  {PASS if ok4 else FAIL} 无 latency_ms 字段 → total_with_latency == 0")


def test_analyze_endpoint_distribution():
    print("\n--- test_analyze_endpoint_distribution ---")

    entries = make_entries([
        {"path": "/api/v1/classify", "level": "info"},
        {"path": "/api/v1/classify", "level": "info"},
        {"path": "/api/v1/classify", "level": "error"},
        {"path": "/api/v1/health", "level": "info"},
    ])
    stats = analyze_endpoint_distribution(entries)

    ok1 = "/api/v1/classify" in stats
    ok2 = stats["/api/v1/classify"]["request_count"] == 3
    ok3 = stats["/api/v1/classify"]["error_count"] == 1
    ok4 = abs(stats["/api/v1/classify"]["error_rate"] - 33.33) < 0.1
    ok5 = stats["/api/v1/health"]["request_count"] == 1

    print(f"  {PASS if ok1 else FAIL} /api/v1/classify 在结果中")
    print(f"  {PASS if ok2 else FAIL} classify 请求数 == 3，实际 {stats['/api/v1/classify']['request_count']}")
    print(f"  {PASS if ok3 else FAIL} classify 错误数 == 1，实际 {stats['/api/v1/classify']['error_count']}")
    print(f"  {PASS if ok4 else FAIL} classify 错误率 ≈ 33.33%，实际 {stats['/api/v1/classify']['error_rate']}%")
    print(f"  {PASS if ok5 else FAIL} health 请求数 == 1，实际 {stats['/api/v1/health']['request_count']}")


def test_analyze_requests_per_minute():
    print("\n--- test_analyze_requests_per_minute ---")

    # 生成跨越 3 分钟的日志
    entries = []
    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    # 第 0 分钟：3 条
    for i in range(3):
        entries.append({"timestamp": (base + timedelta(seconds=i * 10)).isoformat()})
    # 第 1 分钟：5 条（峰值）
    for i in range(5):
        entries.append({"timestamp": (base + timedelta(minutes=1, seconds=i * 10)).isoformat()})
    # 第 2 分钟：2 条
    for i in range(2):
        entries.append({"timestamp": (base + timedelta(minutes=2, seconds=i * 10)).isoformat()})

    stats = analyze_requests_per_minute(entries)

    ok1 = len(stats["time_series"]) == 3
    ok2 = stats["peak_count"] == 5

    print(f"  {PASS if ok1 else FAIL} 时间序列长度 == 3，实际 {len(stats['time_series'])}")
    print(f"  {PASS if ok2 else FAIL} 峰值 == 5，实际 {stats['peak_count']}")


def test_collect_recent_errors():
    print("\n--- test_collect_recent_errors ---")

    entries = make_entries(
        [{"level": "info"}] * 5
        + [{"level": "error", "event": "failure", "path": "/api/v1/classify"}] * 3
    )
    errors = collect_recent_errors(entries, max_errors=2)

    ok1 = len(errors) == 2
    ok2 = all(e["event"] == "failure" for e in errors)

    print(f"  {PASS if ok1 else FAIL} max_errors=2 → 返回 2 条，实际 {len(errors)}")
    print(f"  {PASS if ok2 else FAIL} 所有错误样本 event == 'failure'")


def test_filter_by_time_window():
    print("\n--- test_filter_by_time_window ---")

    base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    entries = [
        {"timestamp": (base - timedelta(minutes=20)).isoformat(), "level": "info"},  # 20分前
        {"timestamp": (base - timedelta(minutes=5)).isoformat(), "level": "info"},   # 5分前
        {"timestamp": base.isoformat(), "level": "info"},                             # 最新
    ]

    # 只保留最近 10 分钟
    filtered = filter_by_time_window(entries, time_window_minutes=10)
    ok = len(filtered) == 2  # 20分前的那条被过滤掉
    print(f"  {PASS if ok else FAIL} 最近 10 分钟 → 期望 2 条，实际 {len(filtered)} 条")

    # time_window=None 不过滤
    all_entries = filter_by_time_window(entries, time_window_minutes=None)
    ok2 = len(all_entries) == 3
    print(f"  {PASS if ok2 else FAIL} time_window=None → 返回全部 3 条，实际 {len(all_entries)}")


if __name__ == "__main__":
    print("=" * 50)
    print("log_analyzer.py 功能验证")
    print("=" * 50)

    test_parse_timestamp()
    test_parse_log_file()
    test_analyze_error_rate()
    test_analyze_slow_requests()
    test_analyze_endpoint_distribution()
    test_analyze_requests_per_minute()
    test_collect_recent_errors()
    test_filter_by_time_window()

    print("\n" + "=" * 50)
    print("所有测试完成")
    print("=" * 50)
