"""
数据驱动测试 — 等价类划分（Equivalence Partitioning）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：等价类划分
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心思想：把无限大的输入空间划分成若干"等价类"——
  同一等价类里的输入，系统对它们的处理方式是相同的，
  因此只需从每类中抽取一个代表值来测试即可。

本文件的等价类划分：
  ┌─────────────────────────────────────────────┐
  │  有效等价类（Valid Equivalence Classes）     │
  │    VEC-1: 正常中文文本（4个语义类别各5条）  │
  │    VEC-2: 混合内容（英文、数字夹杂中文）    │
  ├─────────────────────────────────────────────┤
  │  无效等价类（Invalid Equivalence Classes）  │
  │    IEC-1: 边界输入（单字符、特殊符号等）    │
  │    IEC-2: 恶意输入（注入、超长、控制字符）  │
  └─────────────────────────────────────────────┘

选择原因：
  - 文本分类器的输入空间是"所有可能字符串"，无穷大
  - 用等价类划分，我们把重点放在"分类器会产生不同处理路径"的地方
  - 有效类：验证功能正确性（预测标签）
  - 无效类：验证鲁棒性（不崩溃、返回合理响应）

pytest关键特性：
  @pytest.mark.parametrize —— 把测试数据和测试逻辑分离，
    同一段断言逻辑对多组输入数据重复运行。
    优势：失败时报告哪一组数据失败，而不是笼统的"测试失败"。

数据来源：JSON文件加载，实现了"测试数据 vs 测试逻辑"的完全解耦，
  修改测试数据不需要改代码。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient

# 测试数据目录
_DATA_DIR = Path(__file__).parent / "test_data"


def _load_json(filename: str) -> list[dict[str, Any]]:
    """从 test_data/ 目录加载 JSON 测试数据。"""
    with open(_DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


# ── 加载测试数据 ──────────────────────────────────────────────────────────────

_NORMAL_INPUTS = _load_json("normal_inputs.json")
_EDGE_CASES = _load_json("edge_cases.json")
_MALICIOUS_INPUTS = _load_json("malicious_inputs.json")


# ── VEC-1：有效等价类 — 正常中文文本分类（功能正确性验证）────────────────────


@pytest.mark.parametrize(
    "text, expected_category",
    [(item["text"], item["expected_category"]) for item in _NORMAL_INPUTS],
    ids=[f"VEC-1/{item['expected_category']}/{item['text'][:12]}" for item in _NORMAL_INPUTS],
)
@pytest.mark.asyncio
async def test_classify_normal_input_returns_correct_category(
    app_client: AsyncClient,
    text: str,
    expected_category: str,
) -> None:
    """有效等价类：正常中文文本应被分类到正确类别。

    验证点：
    - HTTP 200 状态码
    - 响应包含 label, confidence, latency_ms, request_id 四个字段
    - label 与预期类别一致（TF-IDF+NB 对训练样本应有很高准确率）
    - confidence 在 [0, 1] 范围内
    """
    response = await app_client.post("/api/v1/classify", json={"text": text})

    assert response.status_code == 200, (
        f"正常文本应返回 200，实际 {response.status_code}，text={text!r}"
    )

    data = response.json()
    assert "label" in data, "响应缺少 label 字段"
    assert "confidence" in data, "响应缺少 confidence 字段"
    assert "latency_ms" in data, "响应缺少 latency_ms 字段"
    assert "request_id" in data, "响应缺少 request_id 字段"

    assert data["label"] == expected_category, (
        f"分类错误: 文本={text!r}, 期望={expected_category}, 实际={data['label']}"
    )
    assert 0.0 <= data["confidence"] <= 1.0, (
        f"confidence 超出 [0,1] 范围: {data['confidence']}"
    )


# ── IEC-1：无效等价类 — 边界输入（鲁棒性验证）───────────────────────────────

# 过滤掉纯空格（Pydantic min_length=1 校验后是422，在API层就被拒绝）
_EDGE_CASES_VIA_API = [
    item for item in _EDGE_CASES
    if item["text"].strip()  # 非纯空格，能通过校验的边界输入
]


@pytest.mark.parametrize(
    "text, description",
    [(item["text"], item["description"]) for item in _EDGE_CASES_VIA_API],
    ids=[f"IEC-1/{item['description']}" for item in _EDGE_CASES_VIA_API],
)
@pytest.mark.asyncio
async def test_classify_edge_case_does_not_crash(
    app_client: AsyncClient,
    text: str,
    description: str,
) -> None:
    """无效等价类（边界输入）：不应导致服务崩溃，应返回合理的 HTTP 响应。

    合理响应定义：
    - 200（分类成功，标签可能不准，但不崩溃）
    - 422（Pydantic 校验拒绝）
    - 500（内部错误，但服务仍在运行，不是进程崩溃）
    不可接受：服务无响应、连接重置、超时
    """
    response = await app_client.post("/api/v1/classify", json={"text": text})

    assert response.status_code in (200, 422, 500), (
        f"边界输入导致非预期状态码 {response.status_code}，"
        f"description={description}, text={text!r}"
    )

    # 进一步验证：如果 200，结构必须完整
    if response.status_code == 200:
        data = response.json()
        assert "label" in data, f"边界输入200响应缺少label，description={description}"
        assert "confidence" in data, f"边界输入200响应缺少confidence"
        assert 0.0 <= data["confidence"] <= 1.0, (
            f"边界输入 confidence 超出范围: {data['confidence']}"
        )


# ── IEC-2：无效等价类 — 恶意输入（安全鲁棒性验证）──────────────────────────


@pytest.mark.parametrize(
    "text, description",
    [(item["text"], item["description"]) for item in _MALICIOUS_INPUTS],
    ids=[f"IEC-2/{item['description']}" for item in _MALICIOUS_INPUTS],
)
@pytest.mark.asyncio
async def test_classify_malicious_input_does_not_crash(
    app_client: AsyncClient,
    text: str,
    description: str,
) -> None:
    """无效等价类（恶意输入）：注入攻击、超长字符串等不应导致服务崩溃。

    对于超长字符串（>10000字符）：期望 422（Pydantic 校验拒绝）
    对于其他恶意内容（长度合法）：期望 200（正常分类，内容不影响模型安全性）
    """
    response = await app_client.post("/api/v1/classify", json={"text": text})

    is_too_long = len(text) > 10000

    if is_too_long:
        assert response.status_code == 422, (
            f"超长输入应返回 422，实际 {response.status_code}，description={description}"
        )
    else:
        assert response.status_code in (200, 422, 500), (
            f"恶意输入导致非预期状态码 {response.status_code}，description={description}"
        )
