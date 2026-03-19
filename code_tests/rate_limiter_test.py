import sys
sys.path.insert(0, ".")

from src.middleware.rate_limiter import TokenBucketRateLimiter
import time

# ── 测试1：基本允许/拒绝 ─────────────────────────────────────────────────────
# capacity=3，新客户端满桶，连发3次应全部允许，第4次应被拒绝
limiter = TokenBucketRateLimiter(capacity=3, refill_rate=1.0)
results = [limiter.allow("user_a") for _ in range(4)]
print("测试1 连发4次（capacity=3）:", results)
# 预期: [True, True, True, False]

# ── 测试2：不同 client_id 桶独立 ─────────────────────────────────────────────
# user_b 和 user_c 各自有独立的桶，互不影响
limiter2 = TokenBucketRateLimiter(capacity=1, refill_rate=1.0)
r_b = limiter2.allow("user_b")
r_c = limiter2.allow("user_c")
print("测试2 user_b:", r_b, "user_c:", r_c)
# 预期: True True（各自满桶）

r_b2 = limiter2.allow("user_b")
r_c2 = limiter2.allow("user_c")
print("测试2 再次 user_b:", r_b2, "user_c:", r_c2)
# 预期: False False（各自桶空）

# ── 测试3：等待后令牌补充 ────────────────────────────────────────────────────
# capacity=2, refill_rate=2.0 → 每秒补充2个
# 先耗尽，等0.6秒，应该补充约1.2个，可以再发1次
limiter3 = TokenBucketRateLimiter(capacity=2, refill_rate=2.0)
limiter3.allow("user_d")
limiter3.allow("user_d")  # 耗尽
r_empty = limiter3.allow("user_d")
print("测试3 耗尽后立即请求:", r_empty)
# 预期: False

time.sleep(0.6)
r_refilled = limiter3.allow("user_d")
print("测试3 等待0.6s后请求:", r_refilled)
# 预期: True（0.6s × 2.0/s = 1.2个令牌，>= 1）

# ── 测试4：reset 后恢复满桶 ──────────────────────────────────────────────────
limiter4 = TokenBucketRateLimiter(capacity=1, refill_rate=1.0)
limiter4.allow("user_e")  # 耗尽
r_before = limiter4.allow("user_e")
print("测试4 reset前:", r_before)
# 预期: False

limiter4.reset("user_e")
r_after = limiter4.allow("user_e")
print("测试4 reset后:", r_after)
# 预期: True（重置为满桶）

# ── 测试5：get_bucket_state 返回当前令牌数 ───────────────────────────────────
limiter5 = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
limiter5.allow("user_f")
limiter5.allow("user_f")
state = limiter5.get_bucket_state("user_f")
print("测试5 消耗2个后令牌数:", state["tokens"])
# 预期: 3.0（5 - 2 = 3，时间极短补充可忽略）
