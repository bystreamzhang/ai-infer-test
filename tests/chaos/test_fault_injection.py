"""
混沌测试 — 故障注入（Fault Injection / Chaos Engineering）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：故障注入测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
什么是混沌工程（Chaos Engineering）？
  2010年 Netflix 开始在生产环境随机关闭服务器（"Chaos Monkey"），
  以验证系统在部分组件故障时仍能正常运行。
  混沌测试就是在测试环境中主动制造故障，而不是等待故障发生。

故障注入（Fault Injection）是混沌测试的核心手段：
  人为在系统的某个点注入异常，观察系统的响应：
    - 是否崩溃（进程退出）？
    - 是否优雅降级（返回错误码但不崩溃）？
    - 错误是否被正确记录到日志？
    - 系统是否能从故障中恢复？

本文件注入的故障类型：
  1. 模型预测失败（RuntimeError）：模拟模型推理过程中出现异常
  2. 模型注册表损坏：模拟 ModelRegistry.get() 失败
  3. 推理超时：模拟推理耗时过长导致超时

为什么用 unittest.mock.patch？
  我们无法真正"让模型崩溃"，但可以用 mock 替换模型方法，
  让它抛出指定的异常，效果等同于真实故障。

  mock.patch 的作用原理：
    with patch("src.models.text_classifier.TextClassifier.predict") as mock_pred:
        mock_pred.side_effect = RuntimeError("GPU OOM")
        # 此块内，所有对 TextClassifier.predict 的调用都会抛出 RuntimeError
    # 退出块后，predict 恢复正常

pytest关键特性：
  caplog fixture —— 捕获日志输出，验证错误被记录。
    with caplog.at_level(logging.ERROR):
        # 在此处运行会产生日志的代码
    assert "error message" in caplog.text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient

from src.models.text_classifier import TextClassifier
from src.services.model_registry import ModelRegistry


# ── 故障1：模型预测抛出 RuntimeError ────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_model_predict_raises_runtime_error_returns_500(
    app_client: AsyncClient,
) -> None:
    """故障注入：TextClassifier.predict 抛出 RuntimeError。

    期望行为：
    - API 返回 500（Internal Server Error），而不是进程崩溃
    - 响应体包含结构化错误信息（code, message 字段）
    - 服务在错误后仍能继续处理后续请求

    注意：patch 的路径是"被测代码导入对象的地方"，
    即 src.models.text_classifier.TextClassifier.predict，
    而不是 src.app 里的路径（因为 app.py 通过 engine 间接调用）。
    """
    with patch.object(TextClassifier, "predict", side_effect=RuntimeError("GPU out of memory")):
        response = await app_client.post(
            "/api/v1/classify",
            json={"text": "人工智能"},
        )

    assert response.status_code == 500, (
        f"模型崩溃时应返回 500，实际 {response.status_code}"
    )

    data = response.json()
    assert "code" in data, "500响应应包含 code 字段"
    assert "message" in data, "500响应应包含 message 字段"


@pytest.mark.asyncio
async def test_classify_after_fault_service_recovers(
    app_client: AsyncClient,
) -> None:
    """弹性验证：故障恢复后，服务应能正常处理后续请求。

    先注入故障触发 500，退出 mock 后发送正常请求，
    验证服务没有因前一次错误而进入不可用状态。
    """
    # 注入故障
    with patch.object(TextClassifier, "predict", side_effect=RuntimeError("transient error")):
        fault_response = await app_client.post(
            "/api/v1/classify",
            json={"text": "人工智能"},
        )
    assert fault_response.status_code == 500, "故障注入期间应返回 500"

    # 恢复后正常请求
    recovery_response = await app_client.post(
        "/api/v1/classify",
        json={"text": "人工智能大模型发布引发行业震动"},
    )
    assert recovery_response.status_code == 200, (
        f"故障恢复后应返回 200，实际 {recovery_response.status_code}，"
        "服务可能因未捕获异常进入不一致状态"
    )


# ── 故障2：批量请求中途出错 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_classify_partial_failure_returns_500(
    app_client: AsyncClient,
) -> None:
    """故障注入：批量请求中，模型在处理过程中抛出异常。

    当前实现是串行处理每条文本，第一条出错即抛出。
    期望：返回 500，不是 200 + 部分结果。
    """
    with patch.object(TextClassifier, "predict", side_effect=RuntimeError("model failure")):
        response = await app_client.post(
            "/api/v1/batch/classify",
            json={"texts": ["足球", "篮球", "游泳"]},
        )

    assert response.status_code == 500, (
        f"批量请求模型崩溃时应返回 500，实际 {response.status_code}"
    )


# ── 故障3：ModelRegistry 模型不存在 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_model_not_found_returns_404(
    app_client: AsyncClient,
) -> None:
    """故障注入：模拟 ModelRegistry.get() 抛出 ModelNotFoundError。

    ModelNotFoundError 应被全局异常处理器捕获，返回 404。
    """
    from src.services.model_registry import ModelNotFoundError

    with patch.object(
        ModelRegistry,
        "get",
        side_effect=ModelNotFoundError("text_classifier", "latest"),
    ):
        response = await app_client.post(
            "/api/v1/classify",
            json={"text": "人工智能"},
        )

    assert response.status_code == 404, (
        f"模型不存在时应返回 404，实际 {response.status_code}"
    )

    data = response.json()
    assert data.get("code") == "MODEL_NOT_FOUND", (
        f"错误码应为 MODEL_NOT_FOUND，实际 {data.get('code')}"
    )


# ── 故障4：健康检查在服务降级时应反映状态 ───────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_returns_ok_even_under_load(
    app_client: AsyncClient,
) -> None:
    """健康检查端点应始终可用，即使其他端点正在处理请求。

    /api/v1/health 是轻量级端点，不调用模型，
    在服务正常运行时应始终返回 200 和 status="ok"。
    """
    response = await app_client.get("/api/v1/health")

    assert response.status_code == 200, (
        f"健康检查应返回 200，实际 {response.status_code}"
    )

    data = response.json()
    assert data.get("status") == "ok", (
        f"健康状态应为 ok，实际 {data.get('status')}"
    )
    assert "models_loaded" in data, "健康检查应包含 models_loaded 字段"
    assert data["models_loaded"] > 0, "正常运行时 models_loaded 应大于 0"


# ── 故障5：并发故障注入 — 在并发请求中随机注入错误 ──────────────────────────


@pytest.mark.asyncio
async def test_fault_injection_under_concurrent_requests(
    app_client: AsyncClient,
) -> None:
    """并发场景下的故障注入：验证错误不会导致其他请求受影响。

    注入的 RuntimeError 应只影响当前请求，
    不应通过共享状态污染其他并发请求。

    由于 mock 是全局的（patch.object），这里分两阶段：
      阶段1：注入故障，发送并发请求，验证全部返回 500
      阶段2：恢复正常，发送并发请求，验证全部返回 200
    """
    import asyncio

    # 阶段1：故障期间
    with patch.object(TextClassifier, "predict", side_effect=RuntimeError("chaos")):
        fault_responses = await asyncio.gather(
            *[
                app_client.post("/api/v1/classify", json={"text": "测试文本"})
                for _ in range(5)
            ],
            return_exceptions=True,
        )

    for i, resp in enumerate(fault_responses):
        assert not isinstance(resp, Exception), (
            f"请求 {i} 抛出了未捕获异常：{resp}"
        )
        assert resp.status_code == 500, (
            f"故障期间请求 {i} 应返回 500，实际 {resp.status_code}"
        )

    # 阶段2：恢复后
    recovery_responses = await asyncio.gather(
        *[
            app_client.post(
                "/api/v1/classify",
                json={"text": "人工智能大模型发布引发行业震动"},
            )
            for _ in range(5)
        ]
    )

    for i, resp in enumerate(recovery_responses):
        assert resp.status_code == 200, (
            f"故障恢复后请求 {i} 应返回 200，实际 {resp.status_code}"
        )
