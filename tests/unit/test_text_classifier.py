"""
单元测试：TextClassifier

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：等价类划分（Equivalence Partitioning）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
等价类划分的核心思想：
  把输入空间划分为若干"等价类"，同一等价类内的输入对被测系统的行为
  是等价的（要么都通过，要么都失败）。每类只取一个代表测试即可。

本文件的等价类划分：
  ┌──────────────────────────────────────────┐
  │ 等价类          │ 代表性输入              │
  ├──────────────────────────────────────────┤
  │ 正常文本（体育）  │ "中国队赢得世界杯…"     │
  │ 正常文本（科技）  │ "人工智能大模型…"       │
  │ 正常文本（娱乐）  │ "国产电影票房…"         │
  │ 正常文本（财经）  │ "A股市场成交量…"        │
  │ 空字符串         │ ""                     │
  │ 超长文本         │ "a" * 10000            │
  │ 特殊字符         │ "<script>", emoji      │
  └──────────────────────────────────────────┘

pytest 关键特性：
  @pytest.mark.parametrize  — 用数据表驱动同一测试逻辑，避免重复代码
  fixture（classifier_model） — session scope，只训练一次模型（~200ms）
"""

import pytest

from src.models.text_classifier import TextClassifier


# ── 参数化数据 ────────────────────────────────────────────────────────────────

# 已知样本：从训练数据中取，模型应能正确分类
KNOWN_SAMPLES = [
    ("中国队赢得世界杯小组赛首胜", "sports"),
    ("NBA总决赛湖人队夺冠", "sports"),
    ("人工智能大模型发布引发行业震动", "tech"),
    ("量子计算机突破传统计算极限", "tech"),
    ("国产电影票房突破百亿创历史", "entertainment"),
    ("流行歌手新专辑首日播放量破纪录", "entertainment"),
    ("A股市场成交量创年内新高", "finance"),
    ("央行宣布降准释放长期流动性", "finance"),
]

# 不应崩溃的边界/特殊输入
SAFE_INPUTS = [
    "",                                      # 空字符串
    "a",                                     # 单字符
    "a" * 10_000,                            # 超长文本
    "<script>alert('xss')</script>",         # HTML/XSS
    "' OR 1=1 --",                           # SQL 注入
    "🎉🎊🎈🎁🎀",                              # 纯 emoji
    "αβγδεζηθ",                              # 希腊字母
    "   \t\n  ",                             # 纯空白
    "0" * 500,                               # 重复数字
]


# ── 正确性测试 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected_label", KNOWN_SAMPLES)
def test_classify_known_samples_correct_label(
    classifier_model: TextClassifier,
    text: str,
    expected_label: str,
) -> None:
    """训练集中的样本应被正确分类。

    等价类：正常文本（各类别代表）
    """
    result = classifier_model.predict(text)
    assert result["label"] == expected_label, (
        f"Expected '{expected_label}', got '{result['label']}' for text: {text!r}"
    )


# ── 返回值结构校验 ────────────────────────────────────────────────────────────


def test_classify_returns_required_keys(classifier_model: TextClassifier) -> None:
    """predict() 返回值必须包含 label、confidence、latency_ms 三个 key。"""
    result = classifier_model.predict("人工智能技术发展迅速")
    required_keys = {"label", "confidence", "latency_ms"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_classify_confidence_in_range(classifier_model: TextClassifier) -> None:
    """confidence 必须在 [0, 1] 范围内。"""
    result = classifier_model.predict("体育运动健康生活")
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"confidence out of range: {result['confidence']}"
    )


def test_classify_label_is_valid_category(classifier_model: TextClassifier) -> None:
    """label 必须是四个已知类别之一。"""
    valid_labels = {"sports", "tech", "entertainment", "finance"}
    result = classifier_model.predict("今天的比赛非常精彩")
    assert result["label"] in valid_labels, (
        f"Unexpected label: {result['label']!r}"
    )


def test_classify_latency_ms_positive(classifier_model: TextClassifier) -> None:
    """latency_ms 必须是正数。"""
    result = classifier_model.predict("测试延迟")
    assert result["latency_ms"] > 0, (
        f"latency_ms should be positive, got: {result['latency_ms']}"
    )


# ── 鲁棒性测试（不应崩溃）────────────────────────────────────────────────────


@pytest.mark.parametrize("text", SAFE_INPUTS)
def test_classify_safe_inputs_no_crash(
    classifier_model: TextClassifier,
    text: str,
) -> None:
    """边界/特殊输入不应导致崩溃，必须返回合理结果。

    等价类：空字符串、超长文本、特殊字符、emoji
    即使分类结果不准确，也必须：
      1. 不抛出异常
      2. 返回包含必要字段的字典
      3. confidence 在合法范围内
    """
    result = classifier_model.predict(text)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "label" in result, "Missing 'label' key"
    assert "confidence" in result, "Missing 'confidence' key"
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"confidence out of range: {result['confidence']}"
    )


# ── batch 预测测试 ────────────────────────────────────────────────────────────


def test_batch_classify_count_matches_input(classifier_model: TextClassifier) -> None:
    """batch 预测结果数量必须与输入数量一致。"""
    texts = ["体育新闻", "科技资讯", "娱乐八卦", "财经动态"]
    results = classifier_model.predict_batch(texts)
    assert len(results) == len(texts), (
        f"Expected {len(texts)} results, got {len(results)}"
    )


def test_batch_classify_empty_list(classifier_model: TextClassifier) -> None:
    """空列表输入应返回空列表，不崩溃。"""
    results = classifier_model.predict_batch([])
    assert results == [], f"Expected empty list, got: {results}"


def test_batch_classify_single_item(classifier_model: TextClassifier) -> None:
    """单条文本的 batch 与 predict 结果应一致（标签相同）。"""
    text = "中国队赢得世界杯小组赛首胜"
    single_result = classifier_model.predict(text)
    batch_result = classifier_model.predict_batch([text])
    assert len(batch_result) == 1, "batch 应返回 1 条结果"
    assert batch_result[0]["label"] == single_result["label"], (
        "batch 与 predict 标签不一致"
    )


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_batch_classify_various_sizes(
    classifier_model: TextClassifier,
    batch_size: int,
) -> None:
    """不同 batch 大小都应正常处理。"""
    texts = ["人工智能"] * batch_size
    results = classifier_model.predict_batch(texts)
    assert len(results) == batch_size, (
        f"Expected {batch_size} results, got {len(results)}"
    )
