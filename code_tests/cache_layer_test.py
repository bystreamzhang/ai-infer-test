from src.services.cache_layer import LRUTTLCache
import time

# 基本 put/get
cache = LRUTTLCache(max_size=3, ttl_seconds=1.0)
cache.put('k1', 'hello')
print('get k1:', cache.get('k1'))         # 预期: hello
print('get k2:', cache.get('k2'))         # 预期: None（未命中）

# TTL 过期
cache.put('k2', 'world')
time.sleep(1.1)
print('get k2 after TTL:', cache.get('k2'))  # 预期: None（已过期）

# LRU 淘汰：max_size=3，插入第4条时踢掉最旧的
cache2 = LRUTTLCache(max_size=3, ttl_seconds=60.0)
cache2.put('a', 1)
cache2.put('b', 2)
cache2.put('c', 3)
cache2.get('a')          # 访问 a，使它成为最近使用
cache2.put('d', 4)       # 容量满，淘汰最久未用的 b
print('get b:', cache2.get('b'))   # 预期: None（b 被淘汰）
print('get a:', cache2.get('a'))   # 预期: 1（a 被保护）

# 统计
print('stats:', cache2.stats())
# 预期: hit_rate > 0, evictions=1, current_size=3