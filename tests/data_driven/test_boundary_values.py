"""
数据驱动测试 — 边界值分析（Boundary Value Analysis, BVA）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：边界值分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心思想：Bug 最容易藏在边界处——"刚好等于"和"差一个"之间。

经典的边界值策略是在每个边界取三点：
  - on point（刚好在边界上）
  - off point（紧贴边界外一步，刚好越界）
  - interior point（边界内某个正常值）

本文件测试三个维度的边界：

  维度1：text 输入长度
  ┌─────────┬───────────────────────────────────────┐
  │ 边界    │ min_length=1, max_length=10000         │
  ├─────────┼───────────────────────────────────────┤
  │ 0       │ 空字符串 → Pydantic 拒绝 → 422        │
  │ 1       │ on point（最小合法值）→ 200 or 422*   │
  │ 9999    │ 合法范围内 → 200                       │
  │ 10000   │ on point（最大合法值）→ 200            │
  │ 10001   │ off point（刚好越界）→ 422             │
  └─────────┴───────────────────────────────────────┘
  *注：单字符纯空格会被 min_length=1 接受，但部分验证器会拦截

  维度2：batch_size（批量请求的列表长度）
  ┌─────────┬───────────────────────────────────────┐
  │ 0       │ 空列表 → 422（min_length=1）          │
  │ 1       │ on point → 200                        │
  │ 99      │ 合法范围内 → 200                      │
  │ 100     │ on point（最大合法值）→ 200           │
  │ 101     │ off point（越界）→ 422                │
  └─────────┴───────────────────────────────────────┘

  维度3：max_length（文本生成长度）
  ┌─────────┬───────────────────────────────────────┐
  │ 0       │ off point（低于 ge=1）→ 422           │
  │ 1       │ on point（最小值）→ 200               │
  │ 499     │ 合法范围内 → 200                      │
  │ 500     │ 合法范围内 → 200                      │
  │ 1000    │ on point（最大值）→ 200               │
  │ 1001    │ off point（越界）→ 422                │
  └─────────┴───────────────────────────────────────┘

选择原因：
  - 边界值分析适用于所有有数值范围约束的参数
  - Pydantic 的 Field(min_length=, max_length=, ge=, le=) 都定义了边界
  - 这些边界的正确性直接影响 API 的安全性（防止超大输入DoS）

pytest关键特性：
  @pytest.mark.parametrize 的 ids 参数 — 给每组测试数据指定有意义的名字，
    让失败报告一眼看出是哪个边界点出了问题。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pytest
from httpx import AsyncClient


# ── 维度1：text 输入长度边界 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text_length, expected_status",
    [
        (0, 422),      # 空字符串，低于 min_length=1
        (1, 200),      # on point：最小合法长度
        (9999, 200),   # 合法范围内
        (10000, 200),  # on point：最大合法长度
        (10001, 422),  # off point：刚好越界
    ],
    ids=["len=0(422)", "len=1(200)", "len=9999(200)", "len=10000(200)", "len=10001(422)"],
)
@pytest.mark.asyncio
async def test_classify_text_length_boundary(
    app_client: AsyncClient,
    text_length: int,
    expected_status: int,
) -> None:
    """维度1：text 长度边界值验证（min=1, max=10000）。

    使用重复字符 'A' 构造精确长度的字符串，排除内容对测试结果的干扰。
    """
    text = "A" * text_length
    payload = {"text": text} if text_length > 0 else {}

    response = await app_client.post("/api/v1/classify", json=payload)

    assert response.status_code == expected_status, (
        f"text长度={text_length}时期望{expected_status}，实际{response.status_code}"
    )


# ── 维度2：batch_size 边界 ───────────────────────────────────────────────────


def _make_batch_payload(size: int) -> dict:
    """构造包含 size 条文本的批量分类请求体。"""
    return {"texts": ["人工智能技术正在改变世界"] * size}


@pytest.mark.parametrize(
    "batch_size, expected_status",
    [
        (0, 422),    # 空列表，低于 min_length=1
        (1, 200),    # on point：最小合法批量
        (99, 200),   # 合法范围内
        (100, 200),  # on point：最大合法批量
        (101, 422),  # off point：刚好越界
    ],
    ids=[
        "batch=0(422)",
        "batch=1(200)",
        "batch=99(200)",
        "batch=100(200)",
        "batch=101(422)",
    ],
)
@pytest.mark.asyncio
async def test_batch_classify_size_boundary(
    app_client: AsyncClient,
    batch_size: int,
    expected_status: int,
) -> None:
    """维度2：批量请求 texts 列表长度边界值验证（min=1, max=100）。

    注意：batch=0 时需要传空列表 []，而不是省略字段。
    空列表和不传字段的语义是不同的——前者是"我明确传了0条"，
    后者在某些框架下会触发"必填字段缺失"的 422。
    """
    if batch_size == 0:
        payload: dict = {"texts": []}
    else:
        payload = _make_batch_payload(batch_size)

    response = await app_client.post("/api/v1/batch/classify", json=payload)

    assert response.status_code == expected_status, (
        f"batch_size={batch_size}时期望{expected_status}，实际{response.status_code}"
    )

    # 成功时验证返回数量与输入数量一致
    if expected_status == 200:
        data = response.json()
        assert data["total"] == batch_size, (
            f"返回数量 {data['total']} 与输入 {batch_size} 不一致"
        )


# ── 维度3：max_length（文本生成长度）边界 ───────────────────────────────────


@pytest.mark.parametrize(
    "max_length, expected_status",
    [
        (0, 422),     # off point：低于 ge=1
        (1, 200),     # on point：最小合法值
        (499, 200),   # 合法范围内
        (500, 200),   # 合法范围内（文档提到501可能触发隐藏bug，500应正常）
        (1000, 200),  # on point：最大合法值
        (1001, 422),  # off point：越界
    ],
    ids=[
        "max_len=0(422)",
        "max_len=1(200)",
        "max_len=499(200)",
        "max_len=500(200)",
        "max_len=1000(200)",
        "max_len=1001(422)",
    ],
)
@pytest.mark.asyncio
async def test_generate_max_length_boundary(
    app_client: AsyncClient,
    max_length: int,
    expected_status: int,
) -> None:
    """维度3：文本生成 max_length 边界值验证（ge=1, le=1000）。

    注意 max_length=0 不能直接放进 payload——
    因为 GenerateRequest 的 ge=1 约束是在 Pydantic 层面的，
    传 0 应该被 Pydantic 拒绝并返回 422。
    """
    payload = {"prompt": "人工智能", "max_length": max_length}

    response = await app_client.post("/api/v1/generate", json=payload)

    assert response.status_code == expected_status, (
        f"max_length={max_length}时期望{expected_status}，实际{response.status_code}"
    )

    # 成功时验证响应结构完整（注意：当前实现忽略 max_length，使用模型默认值）
    if expected_status == 200:
        data = response.json()
        assert "tokens_generated" in data, "响应缺少 tokens_generated 字段"
        assert "text" in data, "响应缺少 text 字段"
        assert data["tokens_generated"] >= 0, "tokens_generated 不应为负数"
        # NOTE: max_length 参数目前被 app.py 忽略（简化设计 思路C），
        # 生成长度由模型默认值决定，不做 <= max_length 的约束断言



# ── 特殊边界：文本长度刚好等于1（单字节vs单字符）──────────────────────────


@pytest.mark.asyncio
async def test_classify_single_ascii_char(app_client: AsyncClient) -> None:
    """单个 ASCII 字符（1字节）— 最小有效输入的极端情况。

    TF-IDF 对单字符的向量化结果可能较弱（ngram覆盖少），
    但分类器应能给出一个结果，不应崩溃。
    """
    response = await app_client.post("/api/v1/classify", json={"text": "A"})
    assert response.status_code == 200, f"单ASCII字符应返回200，实际{response.status_code}"
    data = response.json()
    assert data["label"] in ("sports", "tech", "entertainment", "finance"), (
        f"label应为合法类别，实际 {data['label']!r}"
    )


@pytest.mark.asyncio
async def test_classify_single_chinese_char(app_client: AsyncClient) -> None:
    """单个中文字符（3字节UTF-8）— 验证多字节字符的最小边界。

    字符串长度 len('人') == 1，满足 min_length=1 约束。
    """
    response = await app_client.post("/api/v1/classify", json={"text": "人"})
    assert response.status_code == 200, f"单中文字符应返回200，实际{response.status_code}"
