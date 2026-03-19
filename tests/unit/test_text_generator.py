"""
单元测试：TextGenerator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：边界值分析（Boundary Value Analysis）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
边界值分析的核心思想：
  错误往往藏在"边界"处（off-by-one、越界、空集合）。
  对于每个输入参数，重点测试：
    - 最小值（min）
    - 最小值+1（min+1）
    - 最大值-1（max-1）
    - 最大值（max）
    - 最大值+1（max+1，若有上限）

max_length 参数的边界分析：
  ┌──────────────────────────────────────────────────────────────┐
  │ 值   │ 含义                    │ 预期行为                     │
  ├──────────────────────────────────────────────────────────────┤
  │  0   │ min，不生成任何字符      │ tokens_generated == 0        │
  │  1   │ min+1，生成一个字符      │ tokens_generated <= 1        │
  │ 100  │ 典型正常值               │ tokens_generated <= 100      │
  │ 500  │ 较大值                   │ tokens_generated <= 500      │
  │ 501  │ 可能触发隐藏 bug 的值    │ tokens_generated <= 501      │
  └──────────────────────────────────────────────────────────────┘

注意：Markov Chain 遇到无后继 N-gram 会提前终止，
      所以 tokens_generated <= max_length，不一定等于 max_length。

pytest 关键特性：
  @pytest.mark.parametrize — 用边界值表驱动同一断言逻辑
"""

import pytest

from src.models.text_generator import TextGenerator


# ── 辅助数据 ──────────────────────────────────────────────────────────────────

VALID_PROMPT = "人工智能技术"


# ── 返回值结构校验 ────────────────────────────────────────────────────────────


def test_generate_returns_required_keys(generator_model: TextGenerator) -> None:
    """generate() 返回值必须包含 text、tokens_generated、latency_ms 三个 key。"""
    result = generator_model.generate(VALID_PROMPT, max_length=50)
    required_keys = {"text", "tokens_generated", "latency_ms"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_generate_text_starts_with_prompt(generator_model: TextGenerator) -> None:
    """生成文本应以 prompt 开头（generate() 的语义是"续写"）。

    注：当 prompt 长度 < order=2 时，会被内部替换为语料开头，此处用足够长的 prompt。
    """
    result = generator_model.generate(VALID_PROMPT, max_length=50)
    assert result["text"].startswith(VALID_PROMPT), (
        f"Generated text should start with prompt. "
        f"Got: {result['text'][:30]!r}"
    )


def test_generate_latency_ms_positive(generator_model: TextGenerator) -> None:
    """latency_ms 必须是正数。"""
    result = generator_model.generate(VALID_PROMPT, max_length=10)
    assert result["latency_ms"] > 0, f"latency_ms should be positive"


# ── max_length 边界值测试 ─────────────────────────────────────────────────────


def test_generate_max_length_zero_no_tokens(generator_model: TextGenerator) -> None:
    """max_length=0 时，tokens_generated 应为 0（不生成任何新字符）。"""
    result = generator_model.generate(VALID_PROMPT, max_length=0)
    assert result["tokens_generated"] == 0, (
        f"Expected 0 tokens with max_length=0, got {result['tokens_generated']}"
    )


@pytest.mark.parametrize("max_length", [1, 100, 500, 501])
def test_generate_tokens_not_exceed_max_length(
    generator_model: TextGenerator,
    max_length: int,
) -> None:
    """tokens_generated 不能超过 max_length（边界值：1/100/500/501）。"""
    result = generator_model.generate(VALID_PROMPT, max_length=max_length)
    assert result["tokens_generated"] <= max_length, (
        f"tokens_generated={result['tokens_generated']} exceeded max_length={max_length}"
    )


def test_generate_max_length_one_at_most_one_token(generator_model: TextGenerator) -> None:
    """max_length=1 时，最多生成 1 个字符。"""
    result = generator_model.generate(VALID_PROMPT, max_length=1)
    assert result["tokens_generated"] <= 1, (
        f"Expected at most 1 token with max_length=1, got {result['tokens_generated']}"
    )


def test_generate_tokens_generated_matches_text_length(
    generator_model: TextGenerator,
) -> None:
    """tokens_generated 应等于 len(text) - len(prompt)。"""
    prompt = VALID_PROMPT
    result = generator_model.generate(prompt, max_length=50)
    expected_tokens = len(result["text"]) - len(prompt)
    assert result["tokens_generated"] == expected_tokens, (
        f"tokens_generated={result['tokens_generated']} != "
        f"len(text)-len(prompt)={expected_tokens}"
    )


# ── prompt 边界测试 ───────────────────────────────────────────────────────────


def test_generate_empty_prompt_no_crash(generator_model: TextGenerator) -> None:
    """空 prompt 不应崩溃（内部会补全到 order 个字符）。"""
    result = generator_model.generate("", max_length=20)
    assert isinstance(result, dict), "Should return dict even with empty prompt"
    assert "text" in result, "Result should contain 'text' key"
    assert result["tokens_generated"] <= 20, "tokens_generated should not exceed max_length"


def test_generate_short_prompt_no_crash(generator_model: TextGenerator) -> None:
    """单字符 prompt（短于 order=2）不应崩溃。"""
    result = generator_model.generate("人", max_length=20)
    assert isinstance(result, dict), "Should return dict with short prompt"


def test_generate_large_max_length_terminates(generator_model: TextGenerator) -> None:
    """max_length=500 时不应无限循环，应在合理时间内结束（Markov Chain 会遇到死路）。"""
    result = generator_model.generate(VALID_PROMPT, max_length=500)
    # 关键：函数必须正常返回，tokens_generated <= 500
    assert result["tokens_generated"] <= 500, (
        f"tokens_generated exceeded max_length=500"
    )


# ── predict 接口测试（与 generate 等价）──────────────────────────────────────


def test_predict_is_alias_for_generate(generator_model: TextGenerator) -> None:
    """predict(text) 应等价于 generate(text)（返回值结构相同）。"""
    result = generator_model.predict(VALID_PROMPT)
    required_keys = {"text", "tokens_generated", "latency_ms"}
    assert required_keys.issubset(result.keys()), (
        f"predict() missing keys: {required_keys - result.keys()}"
    )
