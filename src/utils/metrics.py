"""轻量级指标采集模块。

不依赖 prometheus_client，自己实现 Counter / Gauge / Histogram 三种指标，
并提供全局 MetricsRegistry 以 dict 格式导出所有指标快照。
"""

import threading
from typing import Any


class Counter:
    """只增不减的计数器。

    适用场景：请求总数、错误总数、缓存命中次数等"发生了多少次"的统计。

    Example:
        >>> c = Counter("request_total", "总请求数")
        >>> c.inc()
        >>> c.inc(5)
        >>> c.value
        6
    """

    def __init__(self, name: str, description: str = "") -> None:
        """初始化计数器。

        Args:
            name: 指标名称，全局唯一
            description: 指标说明文字
        """
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        """增加计数。

        Args:
            amount: 增加的数量，必须 >= 0

        Raises:
            ValueError: 如果 amount < 0
        """
        if amount < 0:
            raise ValueError(f"Counter increment must be non-negative, got {amount}")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """当前计数值（线程安全读取）。"""
        with self._lock:
            return self._value

    def snapshot(self) -> dict[str, Any]:
        """导出当前状态的快照。

        Returns:
            包含 name, type, value, description 的字典
        """
        return {
            "name": self.name,
            "type": "counter",
            "value": self.value,
            "description": self.description,
        }


class Gauge:
    """可任意增减的仪表盘。

    适用场景：当前并发请求数、队列长度、内存占用等"当前状态值"统计。

    Example:
        >>> g = Gauge("active_requests", "当前活跃请求数")
        >>> g.set(10)
        >>> g.inc(2)
        >>> g.dec(3)
        >>> g.value
        9
    """

    def __init__(self, name: str, description: str = "") -> None:
        """初始化仪表盘。

        Args:
            name: 指标名称，全局唯一
            description: 指标说明文字
        """
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """直接设置当前值。

        Args:
            value: 要设置的值
        """
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """增加指定量。

        Args:
            amount: 增加量（可为负数，等价于 dec）
        """
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """减少指定量。

        Args:
            amount: 减少量（必须 >= 0）
        """
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """当前值（线程安全读取）。"""
        with self._lock:
            return self._value

    def snapshot(self) -> dict[str, Any]:
        """导出当前状态的快照。

        Returns:
            包含 name, type, value, description 的字典
        """
        return {
            "name": self.name,
            "type": "gauge",
            "value": self.value,
            "description": self.description,
        }


class Histogram:
    """记录数值分布的直方图，支持 P50/P95/P99 百分位计算。

    适用场景：请求延迟分布、响应体大小分布等"分布特征"统计。
    内部用一个列表存储所有观测值，百分位查询时排序计算。

    Example:
        >>> h = Histogram("latency_ms", "推理延迟（毫秒）")
        >>> for v in [10, 20, 30, 40, 50]:
        ...     h.observe(v)
        >>> h.percentile(50)
        30.0
    """

    def __init__(self, name: str, description: str = "") -> None:
        """初始化直方图。

        Args:
            name: 指标名称，全局唯一
            description: 指标说明文字
        """
        self.name = name
        self.description = description
        self._values: list[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """记录一个观测值。

        Args:
            value: 观测到的数值（例如一次请求的延迟毫秒数）
        """
        # TODO: 在 _lock 的保护下，将 value 追加到 self._values
        # 提示：with self._lock: 然后 append
        with self._lock:
            self._values.append(value)

    def percentile(self, p: float) -> float:
        """计算指定百分位数。

        P50 = 中位数；P95 = 95% 请求低于此值；P99 = 99% 请求低于此值。

        Args:
            p: 百分位数，范围 0-100（例如传入 95 表示 P95）

        Returns:
            对应百分位的数值；如果没有观测值则返回 0.0

        以下三种实现，哪一种是正确的？

        选项A：
            sorted_vals = sorted(self._values)
            index = int(len(sorted_vals) * p / 100)
            return float(sorted_vals[index])

        选项B：
            sorted_vals = sorted(self._values)
            index = int(len(sorted_vals) * p / 100)
            index = min(index, len(sorted_vals) - 1)
            return float(sorted_vals[index])

        选项C：
            sorted_vals = sorted(self._values)
            index = round(len(sorted_vals) * p / 100)
            return float(sorted_vals[index])

        （提示：考虑 p=100 时的边界情况，以及 round vs int 的区别）
        """
        # TODO: 在 _lock 保护下，先检查 _values 是否为空，再计算百分位
        # 从上面三个选项中选择正确的一个实现
        with self._lock:
            if not self._values:
                return 0.0
            sorted_vals = sorted(self._values)
        index = int(len(sorted_vals) * p / 100)
        index = min(index, len(sorted_vals) - 1)
        return float(sorted_vals[index])

    def stats(self) -> dict[str, float]:
        """返回常用统计量。

        Returns:
            包含 count, min, max, mean, p50, p95, p99 的字典；
            如果没有观测值则全部为 0.0
        """
        with self._lock:
            if not self._values:
                return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
            # 注意：这里直接操作 _values，不需要再获取锁（已经持有）
            # 但是 percentile() 方法内部也会获取锁，会造成死锁！
            # 所以这里直接内联百分位计算，不调用 self.percentile()
            sorted_vals = sorted(self._values)
            n = len(sorted_vals)

            def _p(pct: float) -> float:
                idx = min(int(n * pct / 100), n - 1)
                return float(sorted_vals[idx])

            return {
                "count": float(n),
                "min": float(sorted_vals[0]),
                "max": float(sorted_vals[-1]),
                "mean": float(sum(sorted_vals) / n),
                "p50": _p(50),
                "p95": _p(95),
                "p99": _p(99),
            }

    def snapshot(self) -> dict[str, Any]:
        """导出当前状态的快照。

        Returns:
            包含 name, type, stats, description 的字典
        """
        return {
            "name": self.name,
            "type": "histogram",
            "stats": self.stats(),
            "description": self.description,
        }


class MetricsRegistry:
    """全局指标注册表，统一管理所有 Counter / Gauge / Histogram 实例。

    使用单例模式——整个应用共享同一个 registry 实例。
    通过 snapshot() 将所有指标一次性导出为 dict，供 /metrics 接口返回。

    Example:
        >>> registry = MetricsRegistry()
        >>> c = registry.counter("req_total", "总请求数")
        >>> c.inc()
        >>> registry.snapshot()
        {'req_total': {'name': 'req_total', 'type': 'counter', 'value': 1.0, ...}}
    """

    def __init__(self) -> None:
        """初始化注册表，内部用 dict 按名称索引所有指标。"""
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "") -> Counter:
        """获取或创建一个 Counter。

        如果同名 Counter 已存在则直接返回，否则新建并注册。
        这种"get or create"语义方便在多处代码中引用同一个指标。

        Args:
            name: 指标名称
            description: 指标说明

        Returns:
            Counter 实例
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description)
            return self._metrics[name]  # type: ignore[return-value]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """获取或创建一个 Gauge。

        Args:
            name: 指标名称
            description: 指标说明

        Returns:
            Gauge 实例
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description)
            return self._metrics[name]  # type: ignore[return-value]

    def histogram(self, name: str, description: str = "") -> Histogram:
        """获取或创建一个 Histogram。

        Args:
            name: 指标名称
            description: 指标说明

        Returns:
            Histogram 实例
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description)
            return self._metrics[name]  # type: ignore[return-value]

    def snapshot(self) -> dict[str, Any]:
        """导出所有指标的当前快照。

        Returns:
            以指标名称为 key、各指标 snapshot() 结果为 value 的字典
        """
        with self._lock:
            metrics_copy = dict(self._metrics)
        # 在锁外调用 snapshot()，避免持锁时间过长
        return {name: metric.snapshot() for name, metric in metrics_copy.items()}


# 全局单例，整个应用通过 import 直接使用这个实例
registry = MetricsRegistry()
