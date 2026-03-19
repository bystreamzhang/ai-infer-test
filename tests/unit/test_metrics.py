"""
`pytest tests/unit/test_metrics.py -v`
测试方法论说明
==============

**测试对象**：src/utils/metrics.py 中的 Counter / Gauge / Histogram / MetricsRegistry

**选用的方法论**：

1. 等价类划分（Equivalence Partitioning）
   - Counter.inc：合法输入（正数、0、默认值1）vs 非法输入（负数）
   - Histogram.percentile：有数据 vs 空数据；p=0 vs p=50 vs p=100（边界）

2. 边界值分析（Boundary Value Analysis）
   - percentile(0) / percentile(100)：最小/最大边界
   - 单元素列表：所有百分位应返回同一个值
   - Counter.inc(0)：零值边界（合法，不应抛出）

3. 状态转换测试（State Transition）
   - Gauge：set → inc → dec → set 各种顺序的状态变化
   - Registry："get or create"语义：第一次创建 vs 再次获取应返回同一对象

4. 并发安全测试（Concurrency Testing）
   - 多线程同时 inc / observe，验证最终结果无竞态
   - 这是本模块最关键的非功能需求

**pytest 关键特性**：

- `@pytest.fixture`：在每个测试前创建干净的实例，避免测试间状态污染
  （metrics 有全局状态，fixture 确保隔离）
- `@pytest.mark.parametrize`：用一组参数表达多个等价类，避免重复代码
- `pytest.raises`：断言特定异常被抛出，验证错误处理路径
"""

import threading

import pytest

from src.utils.metrics import Counter, Gauge, Histogram, MetricsRegistry


# ---------------------------------------------------------------------------
# Fixtures：每个测试函数都会得到全新的实例
# ---------------------------------------------------------------------------


@pytest.fixture
def counter() -> Counter:
    """提供一个干净的 Counter 实例。"""
    return Counter("test_counter", "测试用计数器")


@pytest.fixture
def gauge() -> Gauge:
    """提供一个干净的 Gauge 实例。"""
    return Gauge("test_gauge", "测试用仪表盘")


@pytest.fixture
def histogram() -> Histogram:
    """提供一个干净的 Histogram 实例。"""
    return Histogram("test_histogram", "测试用直方图")


@pytest.fixture
def registry() -> MetricsRegistry:
    """提供一个干净的 MetricsRegistry 实例（不使用全局单例）。"""
    return MetricsRegistry()


# ---------------------------------------------------------------------------
# Counter 测试
# ---------------------------------------------------------------------------


class TestCounter:
    def test_counter_initial_value_is_zero(self, counter: Counter) -> None:
        assert counter.value == 0.0, "Counter 初始值应为 0"

    def test_counter_inc_default_increments_by_one(self, counter: Counter) -> None:
        counter.inc()
        assert counter.value == 1.0, "默认 inc() 应增加 1"

    def test_counter_inc_by_amount(self, counter: Counter) -> None:
        counter.inc(5.0)
        assert counter.value == 5.0, "inc(5) 后值应为 5"

    def test_counter_inc_zero_is_allowed(self, counter: Counter) -> None:
        counter.inc(0)
        assert counter.value == 0.0, "inc(0) 是合法操作，值不应变化"

    def test_counter_inc_negative_raises(self, counter: Counter) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            counter.inc(-1)

    def test_counter_accumulates_multiple_incs(self, counter: Counter) -> None:
        for _ in range(10):
            counter.inc()
        assert counter.value == 10.0, "连续 inc 10 次后值应为 10"

    def test_counter_snapshot_structure(self, counter: Counter) -> None:
        counter.inc(3)
        snap = counter.snapshot()
        assert snap["type"] == "counter", "snapshot type 应为 counter"
        assert snap["value"] == 3.0, "snapshot value 应与当前值一致"
        assert snap["name"] == "test_counter", "snapshot name 应与初始化一致"

    def test_counter_thread_safety(self, counter: Counter) -> None:
        """100 个线程各 inc 100 次，最终值应精确等于 10000。"""
        NUM_THREADS = 100
        INC_PER_THREAD = 100

        def worker() -> None:
            for _ in range(INC_PER_THREAD):
                counter.inc()

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == NUM_THREADS * INC_PER_THREAD, (
            f"并发 inc 后值应为 {NUM_THREADS * INC_PER_THREAD}，实际为 {counter.value}"
        )


# ---------------------------------------------------------------------------
# Gauge 测试
# ---------------------------------------------------------------------------


class TestGauge:
    def test_gauge_initial_value_is_zero(self, gauge: Gauge) -> None:
        assert gauge.value == 0.0, "Gauge 初始值应为 0"

    def test_gauge_set(self, gauge: Gauge) -> None:
        gauge.set(42.0)
        assert gauge.value == 42.0, "set 后值应等于设定值"

    def test_gauge_set_can_overwrite(self, gauge: Gauge) -> None:
        gauge.set(10.0)
        gauge.set(3.0)
        assert gauge.value == 3.0, "第二次 set 应覆盖第一次"

    def test_gauge_inc(self, gauge: Gauge) -> None:
        gauge.set(10.0)
        gauge.inc(2.0)
        assert gauge.value == 12.0, "inc 应在当前值基础上增加"

    def test_gauge_dec(self, gauge: Gauge) -> None:
        gauge.set(10.0)
        gauge.dec(3.0)
        assert gauge.value == 7.0, "dec 应在当前值基础上减少"

    def test_gauge_can_go_negative(self, gauge: Gauge) -> None:
        gauge.dec(5.0)
        assert gauge.value == -5.0, "Gauge 可以为负数（不同于 Counter）"

    def test_gauge_snapshot_structure(self, gauge: Gauge) -> None:
        gauge.set(7.0)
        snap = gauge.snapshot()
        assert snap["type"] == "gauge", "snapshot type 应为 gauge"
        assert snap["value"] == 7.0


# ---------------------------------------------------------------------------
# Histogram 测试
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_histogram_empty_percentile_returns_zero(self, histogram: Histogram) -> None:
        assert histogram.percentile(50) == 0.0, "空 Histogram 的百分位应返回 0.0"

    def test_histogram_empty_stats_all_zero(self, histogram: Histogram) -> None:
        s = histogram.stats()
        assert all(v == 0.0 for v in s.values()), "空 Histogram 所有统计量应为 0.0"

    @pytest.mark.parametrize("p", [0, 50, 95, 99, 100])
    def test_histogram_single_value_all_percentiles_equal(
        self, histogram: Histogram, p: float
    ) -> None:
        """只有一个观测值时，所有百分位都应返回该值。"""
        histogram.observe(42.0)
        assert histogram.percentile(p) == 42.0, f"单元素时 P{p} 应等于唯一观测值"

    def test_histogram_p50_with_known_data(self, histogram: Histogram) -> None:
        """[10, 20, 30, 40, 50] 排序后，P50 索引 = int(5*50/100)=2 → 30。"""
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            histogram.observe(v)
        assert histogram.percentile(50) == 30.0, "P50 计算错误"

    def test_histogram_p100_does_not_raise(self, histogram: Histogram) -> None:
        """P100 不应越界（这是 选项A 的 bug，B 修复了它）。"""
        for v in [1.0, 2.0, 3.0]:
            histogram.observe(v)
        result = histogram.percentile(100)
        assert result == 3.0, "P100 应返回最大值"

    def test_histogram_p0_returns_minimum(self, histogram: Histogram) -> None:
        for v in [30.0, 10.0, 20.0]:
            histogram.observe(v)
        assert histogram.percentile(0) == 10.0, "P0 应返回最小值"

    def test_histogram_stats_structure(self, histogram: Histogram) -> None:
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            histogram.observe(v)
        s = histogram.stats()
        assert s["count"] == 5.0
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert s["mean"] == 3.0
        assert "p50" in s and "p95" in s and "p99" in s

    def test_histogram_snapshot_structure(self, histogram: Histogram) -> None:
        histogram.observe(10.0)
        snap = histogram.snapshot()
        assert snap["type"] == "histogram"
        assert "stats" in snap
        assert snap["stats"]["count"] == 1.0

    def test_histogram_thread_safety(self, histogram: Histogram) -> None:
        """50 个线程各 observe 100 次，最终 count 应为 5000。"""
        NUM_THREADS = 50
        OBS_PER_THREAD = 100

        def worker() -> None:
            for i in range(OBS_PER_THREAD):
                histogram.observe(float(i))

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert histogram.stats()["count"] == float(NUM_THREADS * OBS_PER_THREAD), (
            "并发 observe 后 count 应精确等于总观测次数"
        )


# ---------------------------------------------------------------------------
# MetricsRegistry 测试
# ---------------------------------------------------------------------------


class TestMetricsRegistry:
    def test_registry_counter_creates_new(self, registry: MetricsRegistry) -> None:
        c = registry.counter("req_total")
        assert isinstance(c, Counter), "registry.counter 应返回 Counter 实例"

    def test_registry_get_or_create_returns_same_instance(
        self, registry: MetricsRegistry
    ) -> None:
        """同名指标两次获取应返回同一个对象（is 比较，不只是相等）。"""
        c1 = registry.counter("req_total")
        c2 = registry.counter("req_total")
        assert c1 is c2, "同名 Counter 应返回同一个实例"

    def test_registry_snapshot_contains_all_metrics(
        self, registry: MetricsRegistry
    ) -> None:
        registry.counter("c1")
        registry.gauge("g1")
        registry.histogram("h1")
        snap = registry.snapshot()
        assert set(snap.keys()) == {"c1", "g1", "h1"}, "snapshot 应包含所有注册的指标"

    def test_registry_snapshot_reflects_current_values(
        self, registry: MetricsRegistry
    ) -> None:
        c = registry.counter("req")
        c.inc(7)
        snap = registry.snapshot()
        assert snap["req"]["value"] == 7.0, "snapshot 应反映 inc 后的最新值"

    def test_registry_empty_snapshot_is_empty_dict(
        self, registry: MetricsRegistry
    ) -> None:
        assert registry.snapshot() == {}, "空 registry 的 snapshot 应为空字典"

    def test_registry_different_types_same_name_returns_wrong_type(
        self, registry: MetricsRegistry
    ) -> None:
        """先注册为 counter，再用 gauge 获取同名指标——会返回已有的 Counter。
        这是"get or create"语义的隐患，测试记录这一行为（不要求修复）。
        """
        registry.counter("shared_name")
        result = registry.gauge("shared_name")
        # 返回的是已有的 Counter，而不是新的 Gauge
        assert isinstance(result, Counter), (
            "同名已存在时 gauge() 会返回已有的 Counter（已知行为）"
        )
