import sys
sys.path.insert(0, ".")

from pydantic import ValidationError
from src.middleware.request_validator import (
    ClassifyRequest,
    GenerateRequest,
    BatchClassifyRequest,
    ErrorResponse,
)

def ok(label: str, result):
    print(f"[PASS] {label}: {result}")

def expect_error(label: str, exc: ValidationError):
    print(f"[PASS] {label}: 捕获到 ValidationError，字段={[e['loc'] for e in exc.errors()]}")

# ── ClassifyRequest ───────────────────────────────────────────────────────────

# 正常请求
req = ClassifyRequest(text="今天天气不错")
ok("ClassifyRequest 正常", req.text)
# 预期: [PASS] ClassifyRequest 正常: 今天天气不错

# 空字符串 → 报错
try:
    ClassifyRequest(text="")
    print("[FAIL] 空字符串应该报错")
except ValidationError as e:
    expect_error("ClassifyRequest 空字符串", e)
# 预期: [PASS] ... loc=[('text',)]

# 超长文本 → 报错
try:
    ClassifyRequest(text="x" * 10001)
    print("[FAIL] 超长文本应该报错")
except ValidationError as e:
    expect_error("ClassifyRequest 超长文本(10001字符)", e)
# 预期: [PASS] ... loc=[('text',)]

# 边界值：恰好 10000 字符 → 允许
req_max = ClassifyRequest(text="x" * 10000)
ok("ClassifyRequest 边界值10000字符", f"长度={len(req_max.text)}")
# 预期: [PASS] ... 长度=10000

# ── GenerateRequest ───────────────────────────────────────────────────────────

# 正常请求，使用默认 max_length
req2 = GenerateRequest(prompt="从前有座山")
ok("GenerateRequest 默认max_length", f"prompt={req2.prompt}, max_length={req2.max_length}")
# 预期: [PASS] ... max_length=100

# 指定 max_length
req3 = GenerateRequest(prompt="从前有座山", max_length=200)
ok("GenerateRequest 指定max_length=200", req3.max_length)
# 预期: [PASS] ... 200

# max_length 超出范围 → 报错
try:
    GenerateRequest(prompt="test", max_length=1001)
    print("[FAIL] max_length=1001 应该报错")
except ValidationError as e:
    expect_error("GenerateRequest max_length=1001", e)
# 预期: [PASS] ... loc=[('max_length',)]

# max_length=0 → 报错
try:
    GenerateRequest(prompt="test", max_length=0)
    print("[FAIL] max_length=0 应该报错")
except ValidationError as e:
    expect_error("GenerateRequest max_length=0", e)
# 预期: [PASS] ... loc=[('max_length',)]

# ── BatchClassifyRequest ──────────────────────────────────────────────────────

# 正常批量请求
req4 = BatchClassifyRequest(texts=["文本一", "文本二", "文本三"])
ok("BatchClassifyRequest 正常", f"条数={len(req4.texts)}")
# 预期: [PASS] ... 条数=3

# 空列表 → 报错
try:
    BatchClassifyRequest(texts=[])
    print("[FAIL] 空列表应该报错")
except ValidationError as e:
    expect_error("BatchClassifyRequest 空列表", e)
# 预期: [PASS] ... loc=[('texts',)]

# 超过 100 条 → 报错
try:
    BatchClassifyRequest(texts=["x"] * 101)
    print("[FAIL] 101条应该报错")
except ValidationError as e:
    expect_error("BatchClassifyRequest 101条", e)
# 预期: [PASS] ... loc=[('texts',)]

# 单条文本超过 10000 字符 → field_validator 报错
try:
    BatchClassifyRequest(texts=["正常文本", "x" * 10001])
    print("[FAIL] 单条超长应该报错")
except ValidationError as e:
    expect_error("BatchClassifyRequest 单条文本超10000字符", e)
# 预期: [PASS] ... loc=[('texts',)]

# 边界值：恰好 100 条，每条恰好 10000 字符 → 允许
req5 = BatchClassifyRequest(texts=["x" * 10000] * 100)
ok("BatchClassifyRequest 边界值(100条×10000字符)", f"条数={len(req5.texts)}, 首条长度={len(req5.texts[0])}")
# 预期: [PASS] ... 条数=100, 首条长度=10000

# ── ErrorResponse ─────────────────────────────────────────────────────────────

# detail 可选，缺省为 None
err = ErrorResponse(code="MODEL_NOT_FOUND", message="模型不存在")
ok("ErrorResponse 无detail", f"code={err.code}, detail={err.detail}")
# 预期: [PASS] ... detail=None

err2 = ErrorResponse(code="RATE_LIMIT_EXCEEDED", message="请求过快", detail="client=127.0.0.1")
ok("ErrorResponse 有detail", f"detail={err2.detail}")
# 预期: [PASS] ... detail=client=127.0.0.1
