import time
import threading
from collections import OrderedDict
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger("cache_layer")


class LRUTTLCache:
    """LRU + TTL 双策略缓存。

    - LRU（Least Recently Used）：缓存满时淘汰最久未访问的条目。
    - TTL（Time-To-Live）：每条数据有独立的过期时间，过期即视为不存在。
    - 线程安全：所有公共方法均受互斥锁保护。

    内部结构：
        _store: OrderedDict[key, (value, expire_at)]
            左端 = 最久未用（待淘汰），右端 = 最近访问（受保护）
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 60.0) -> None:
        """初始化缓存。

        Args:
            max_size: 最大缓存条目数，超过时淘汰最久未用的条目。
            ttl_seconds: 每条数据的生存时间（秒）。
        """
        self._max_size: int = max_size
        self._ttl: float = ttl_seconds

        # OrderedDict 保存 {key: (value, expire_at)}
        # expire_at 是 time.monotonic() 时间戳，超过即过期
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()

        # 互斥锁，保护 _store 和统计变量
        self._lock: threading.Lock = threading.Lock()

        # 统计变量
        self._total_requests: int = 0
        self._hits: int = 0
        self._evictions: int = 0

        logger.info("LRUTTLCache initialized", max_size=max_size, ttl_seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        """查找缓存。

        若 key 不存在或已过期，返回 None。
        命中时将该条目移到 OrderedDict 末尾（标记为"最近使用"）。

        Args:
            key: 缓存键。

        Returns:
            缓存的值，或 None（未命中/已过期）。
        """
        # TODO 1: 加锁，进入临界区
        # 提示: 用 with self._lock:
        with self._lock:
            self._total_requests += 1

            if key not in self._store:
                logger.debug("cache miss", key=key)
                return None

            value, expire_at = self._store[key]

            # TODO 2: 检查是否已过期
            # 提示: 用 time.monotonic() 获取当前时间，与 expire_at 比较
            # 如果过期：del self._store[key]，记录日志，return None
            if time.monotonic() > expire_at:
                del self._store[key]
                logger.debug("cache expired", key=key)
                return None

            # TODO 3: 命中 — 把该条目移到末尾（表示"最近使用"）
            # 提示: self._store.move_to_end(key)
            self._store.move_to_end(key)

            self._hits += 1
            logger.debug("cache hit", key=key)
            return value

    def put(self, key: str, value: Any) -> None:
        """写入缓存。

        若 key 已存在则更新（并移到末尾）。
        若缓存已满，先淘汰最久未用的条目，再插入新条目。

        Args:
            key: 缓存键。
            value: 要缓存的值。
        """
        expire_at = time.monotonic() + self._ttl

        with self._lock:
            if key in self._store:
                # 已存在：更新值并移到末尾
                self._store[key] = (value, expire_at)
                self._store.move_to_end(key)
                return

            # TODO 4: 容量检查 — 若已满，淘汰最旧的条目
            # 提示: 用 while len(self._store) >= self._max_size:
            #        self._store.popitem(last=False)  # 弹出最左端（最旧）
            #        self._evictions += 1
            while len(self._store) >= self._max_size:
                self._store.popitem(last=False)
                self._evictions += 1

            self._store[key] = (value, expire_at)
            logger.debug("cache put", key=key, expire_at=expire_at)

    def invalidate(self, key: str) -> bool:
        """主动删除指定 key。

        Args:
            key: 要删除的缓存键。

        Returns:
            True 表示成功删除，False 表示 key 不存在。
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug("cache invalidated", key=key)
                return True
            return False

    def stats(self) -> dict[str, Any]:
        """返回缓存运行统计。

        Returns:
            包含以下字段的字典：
                hit_rate: 命中率，范围 [0.0, 1.0]
                total_requests: 总请求次数（get 调用次数）
                hits: 命中次数
                evictions: 因容量满而被淘汰的次数
                current_size: 当前缓存条目数
        """
        with self._lock:
            # TODO 5: 计算命中率
            # 注意：当 total_requests == 0 时需要保护，避免除零错误
            # 提示: hit_rate = self._hits / self._total_requests if self._total_requests > 0 else 0.0
            hit_rate: float = self._hits / self._total_requests if self._total_requests > 0 else 0.0

            return {
                "hit_rate": hit_rate,
                "total_requests": self._total_requests,
                "hits": self._hits,
                "evictions": self._evictions,
                "current_size": len(self._store),
            }
