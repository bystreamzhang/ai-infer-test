"""
单元测试：ModelRegistry

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试方法论：状态转换测试 + 异常路径测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ModelRegistry 是一个典型的"注册表"对象，其行为由内部状态驱动。
测试需要覆盖：

  1. 正常注册流程（CRUD 操作）
     register → get → unregister → get（应失败）

  2. 版本管理逻辑
     多次 register 同名模型 → latest 始终指向最后注册的版本

  3. 异常路径
     get 不存在的模型 → ModelNotFoundError
     unregister 不存在的模型 → ModelNotFoundError
     register/unregister 使用保留字 "latest" → ValueError

  4. 边界情况
     注销最后一个版本 → name 条目应从 _registry 中清除

pytest 关键特性：
  pytest.raises(ExceptionType) — 验证代码抛出预期异常的上下文管理器
  scope="function" fixture — 每个测试独立的 registry（防止注册状态泄漏）
"""

import pytest

from src.services.model_registry import ModelNotFoundError, ModelRegistry


# ── 测试用的假模型对象 ────────────────────────────────────────────────────────

class _FakeModel:
    """最简模型存根，只用于测试注册/获取的对象同一性（identity）。"""
    pass


# ── function scope fixture（每个测试独立注册表）───────────────────────────────

@pytest.fixture
def registry() -> ModelRegistry:
    """每个测试独立的 ModelRegistry 实例。"""
    return ModelRegistry()


# ── 基本注册与获取 ────────────────────────────────────────────────────────────


def test_register_and_get_returns_same_instance(registry: ModelRegistry) -> None:
    """注册后 get() 应返回同一个对象实例（identity check）。"""
    model = _FakeModel()
    registry.register("my_model", "v1.0", model)
    retrieved = registry.get("my_model", "v1.0")
    assert retrieved is model, (
        "get() should return the exact same object that was registered"
    )


def test_register_and_get_latest_returns_same_instance(registry: ModelRegistry) -> None:
    """注册后用 get('name', 'latest') 也应返回同一对象。"""
    model = _FakeModel()
    registry.register("my_model", "v1.0", model)
    retrieved = registry.get("my_model")  # 默认 version="latest"
    assert retrieved is model, "get() with default 'latest' should return registered model"


def test_get_nonexistent_model_raises_error(registry: ModelRegistry) -> None:
    """获取不存在的模型名应抛出 ModelNotFoundError。"""
    with pytest.raises(ModelNotFoundError, match="ghost_model"):
        registry.get("ghost_model")


def test_get_nonexistent_version_raises_error(registry: ModelRegistry) -> None:
    """获取存在的模型名但不存在的版本应抛出 ModelNotFoundError。"""
    model = _FakeModel()
    registry.register("my_model", "v1.0", model)

    with pytest.raises(ModelNotFoundError):
        registry.get("my_model", "v99.0")  # v99.0 从未注册


# ── 多版本与 latest 指针 ──────────────────────────────────────────────────────


def test_latest_points_to_last_registered_version(registry: ModelRegistry) -> None:
    """多版本注册后，latest 应指向最后一次注册的版本。"""
    model_v1 = _FakeModel()
    model_v2 = _FakeModel()

    registry.register("my_model", "v1.0", model_v1)
    registry.register("my_model", "v2.0", model_v2)

    latest = registry.get("my_model", "latest")
    assert latest is model_v2, "latest should point to the most recently registered version"


def test_older_versions_still_accessible_after_new_register(
    registry: ModelRegistry,
) -> None:
    """注册新版本后，旧版本应仍可通过具体版本号访问。"""
    model_v1 = _FakeModel()
    model_v2 = _FakeModel()

    registry.register("my_model", "v1.0", model_v1)
    registry.register("my_model", "v2.0", model_v2)

    assert registry.get("my_model", "v1.0") is model_v1, "v1.0 should still be accessible"
    assert registry.get("my_model", "v2.0") is model_v2, "v2.0 should be accessible"


def test_register_overwrite_existing_version(registry: ModelRegistry) -> None:
    """重复注册同名同版本应覆盖旧实例。"""
    old_model = _FakeModel()
    new_model = _FakeModel()

    registry.register("my_model", "v1.0", old_model)
    registry.register("my_model", "v1.0", new_model)

    assert registry.get("my_model", "v1.0") is new_model, (
        "Re-registering same version should overwrite old instance"
    )


def test_register_reserved_version_raises_error(registry: ModelRegistry) -> None:
    """注册时使用保留字 'latest' 作为版本号应抛出 ValueError。"""
    with pytest.raises(ValueError, match="latest"):
        registry.register("my_model", "latest", _FakeModel())


# ── 注销测试 ──────────────────────────────────────────────────────────────────


def test_unregister_then_get_raises_error(registry: ModelRegistry) -> None:
    """注销后 get() 应抛出 ModelNotFoundError。"""
    model = _FakeModel()
    registry.register("my_model", "v1.0", model)
    registry.unregister("my_model", "v1.0")

    with pytest.raises(ModelNotFoundError):
        registry.get("my_model", "v1.0")


def test_unregister_latest_clears_latest_pointer(registry: ModelRegistry) -> None:
    """注销 latest 指向的版本后，latest 也应失效。"""
    model = _FakeModel()
    registry.register("my_model", "v1.0", model)
    registry.unregister("my_model", "v1.0")

    with pytest.raises(ModelNotFoundError):
        registry.get("my_model", "latest")


def test_unregister_last_version_removes_model_entry(registry: ModelRegistry) -> None:
    """注销最后一个版本后，整个模型名条目应从注册表中清除。"""
    registry.register("my_model", "v1.0", _FakeModel())
    registry.unregister("my_model", "v1.0")

    # 模型名也不应存在了
    with pytest.raises(ModelNotFoundError):
        registry.get("my_model")

    # list_models 中也不应出现
    assert not any(m["name"] == "my_model" for m in registry.list_models()), (
        "Unregistered model should not appear in list_models()"
    )


def test_unregister_nonexistent_raises_error(registry: ModelRegistry) -> None:
    """注销不存在的版本应抛出 ModelNotFoundError。"""
    with pytest.raises(ModelNotFoundError):
        registry.unregister("ghost_model", "v1.0")


def test_unregister_reserved_version_raises_error(registry: ModelRegistry) -> None:
    """用 'latest' 注销应抛出 ValueError。"""
    registry.register("my_model", "v1.0", _FakeModel())
    with pytest.raises(ValueError):
        registry.unregister("my_model", "latest")


def test_unregister_non_latest_version_latest_pointer_intact(
    registry: ModelRegistry,
) -> None:
    """注销旧版本（非 latest 指向的版本）时，latest 不应受影响。"""
    model_v1 = _FakeModel()
    model_v2 = _FakeModel()

    registry.register("my_model", "v1.0", model_v1)
    registry.register("my_model", "v2.0", model_v2)  # latest → v2.0

    registry.unregister("my_model", "v1.0")  # 注销 v1.0，latest 不变

    assert registry.get("my_model", "latest") is model_v2, (
        "latest should still point to v2.0 after v1.0 is unregistered"
    )


# ── list_models 测试 ──────────────────────────────────────────────────────────


def test_list_models_empty_initially(registry: ModelRegistry) -> None:
    """初始状态 list_models() 应返回空列表。"""
    assert registry.list_models() == [], "Empty registry should return empty list"


def test_list_models_returns_correct_entries(registry: ModelRegistry) -> None:
    """list_models() 应返回所有注册模型的元数据（不含 latest）。"""
    registry.register("classifier", "v1.0", _FakeModel())
    registry.register("generator", "v1.0", _FakeModel())

    models = registry.list_models()
    names = [m["name"] for m in models]

    assert len(models) == 2, f"Expected 2 models, got {len(models)}"
    assert "classifier" in names, "classifier should be in list"
    assert "generator" in names, "generator should be in list"


def test_list_models_excludes_latest_virtual_version(registry: ModelRegistry) -> None:
    """list_models() 不应包含 'latest' 虚拟版本条目。"""
    registry.register("my_model", "v1.0", _FakeModel())
    registry.register("my_model", "v2.0", _FakeModel())

    models = registry.list_models()
    versions = [m["version"] for m in models]

    assert "latest" not in versions, "list_models() should not include 'latest' version"
    assert len(models) == 2, f"Expected 2 entries (v1.0 and v2.0), got {len(models)}"


def test_list_models_has_required_fields(registry: ModelRegistry) -> None:
    """list_models() 中每个条目必须有 name、version、created_at 字段。"""
    registry.register("my_model", "v1.0", _FakeModel())
    models = registry.list_models()

    assert len(models) == 1
    entry = models[0]
    assert "name" in entry, "Entry missing 'name'"
    assert "version" in entry, "Entry missing 'version'"
    assert "created_at" in entry, "Entry missing 'created_at'"
