"""模型注册中心：集中管理模型实例与版本。

注册表模式（Registry Pattern）：维护一个中心化的名称->版本->实例映射，
让系统各部分通过统一接口查找模型，避免到处传递模型实例。
"""

import time
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── 自定义异常 ──────────────────────────────────────────────────────────────


class ModelNotFoundError(Exception):
    """当请求的模型名称或版本不存在于注册中心时抛出。"""

    pass


# ── 模型元数据 ──────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    """存储单个模型版本的元数据。

    Attributes:
        name: 模型名称，例如 "text_classifier"。
        version: 版本号，例如 "v1.0"。
        model: 模型实例，可以是任意对象（TextClassifier、TextGenerator 等）。
        created_at: 注册时的 Unix 时间戳（由 field(default_factory=...) 自动填充）。
    """

    name: str
    version: str
    model: Any
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """将元数据序列化为可 JSON 化的字典（不含 model 实例本身）。

        Returns:
            包含 name、version、created_at 的字典。
        """
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
        }


# ── 注册中心 ────────────────────────────────────────────────────────────────


class ModelRegistry:
    """模型注册中心，支持多版本管理与 latest 快捷访问。

    内部存储结构：
        _registry: dict[str, dict[str, ModelInfo]]
        即 { model_name: { version_str: ModelInfo, "latest": ModelInfo } }

    "latest" 是一个特殊 key，始终指向同名模型中最后注册的版本。
    """

    def __init__(self) -> None:
        # TODO: 初始化内部字典 _registry
        # 提示：用普通 dict，结构为 {name: {version: ModelInfo}}
        # "latest" 也作为一个普通的 version key 存在，指向最新注册的 ModelInfo
        self._registry: dict[str, dict[str, ModelInfo]] = {}

        logger.info("model_registry_initialized")

    def register(self, name: str, version: str, model_instance: Any) -> None:
        """注册一个模型版本。

        若同名版本已存在，会覆盖旧实例。
        注册后，"latest" 指针更新为本次注册的版本。

        Args:
            name: 模型名称。
            version: 版本号，不能为 "latest"（保留关键字）。
            model_instance: 模型实例。

        Raises:
            ValueError: 当 version == "latest" 时（保留关键字）。
        """
        if version == "latest":
            raise ValueError('"latest" is a reserved version keyword')

        info = ModelInfo(name=name, version=version, model=model_instance)

        # TODO: 将 info 存入 _registry[name][version]
        # 如果 _registry 中还没有这个 name，需要先创建空字典
        # 伪代码：
        #   if name not in _registry: _registry[name] = {}
        #   _registry[name][version] = info
        if name not in self._registry:
            self._registry[name] = {}
        self._registry[name][version] = info

        # TODO: 更新 "latest" 指针，让 _registry[name]["latest"] 也指向 info
        self._registry[name]["latest"] = info

        logger.info(
            "model_registered",
            model_name=name,
            version=version,
        )

    def get(self, name: str, version: str = "latest") -> Any:
        """按名称和版本获取模型实例。

        Args:
            name: 模型名称。
            version: 版本号，默认 "latest" 获取最新版本。

        Returns:
            模型实例（ModelInfo.model 字段）。

        Raises:
            ModelNotFoundError: 当 name 或 version 不存在时。

        下面有三种写法，其中只有一种能正确处理所有情况，另两种有隐藏bug：

        写法A：
            return self._registry[name][version].model

        写法B：
            if name not in self._registry:
                raise ModelNotFoundError(f"Model '{name}' not found")
            return self._registry[name][version].model

        写法C：
            if name not in self._registry or version not in self._registry[name]:
                raise ModelNotFoundError(f"Model '{name}' version '{version}' not found")
            return self._registry[name][version].model

        思考：哪种写法有什么问题？（不要直接告诉我答案）
        异常处理
        """
        # TODO: 用上面三种写法中正确的那种实现
        if name not in self._registry or version not in self._registry[name]:
            raise ModelNotFoundError(f"Model '{name}' version '{version}' not found")
        return self._registry[name][version].model

    def unregister(self, name: str, version: str) -> None:
        """注销指定版本的模型。

        若被注销的版本恰好是 latest 指向的版本，则 latest 指针也一并清除。
        若该模型名下没有更多版本，整个 name 条目也从 _registry 中删除。

        Args:
            name: 模型名称。
            version: 版本号，不能为 "latest"（用具体版本号来注销）。

        Raises:
            ValueError: 当 version == "latest" 时。
            ModelNotFoundError: 当指定版本不存在时。
        """
        if version == "latest":
            raise ValueError("Use specific version string to unregister")

        # TODO: 检查 name 和 version 是否存在，不存在则抛出 ModelNotFoundError
        if name not in self._registry or version not in self._registry[name]:
            raise ModelNotFoundError(f"Model '{name}' version '{version}' not found")

        # TODO: 删除 _registry[name][version]
        del self._registry[name][version]

        # TODO: 如果被删的版本恰好就是 latest 指向的版本，把 latest 也删掉
        # 提示：比较 _registry[name]["latest"].version == version
        if "latest" in self._registry[name] and self._registry[name]["latest"].version == version:
            del self._registry[name]["latest"]

        # TODO: 如果 name 下已经没有任何版本（只剩 latest key 或全空），
        #       则删除整个 _registry[name] 条目
        # 提示：过滤掉 "latest" key 后，如果 len == 0，就 del _registry[name]
        if len(self._registry[name]) == 0 or (len(self._registry[name]) == 1 and "latest" in self._registry[name]):
            del self._registry[name]
        logger.info("model_unregistered", model_name=name, version=version)

    def list_models(self) -> list[dict]:
        """列出所有已注册模型的元数据（不含 latest 虚拟版本）。

        Returns:
            ModelInfo.to_dict() 的列表，按注册时间升序排列。

        提示：遍历 _registry，跳过 version == "latest" 的条目，
        调用每个 ModelInfo 的 to_dict() 方法收集结果。
        """
        # TODO: 实现遍历逻辑
        # 伪代码：
        #   results = []
        #   for name in _registry:
        #       for version in _registry[name]:
        #           if version != "latest": results.append(...)
        #   return sorted(results, key=lambda x: x["created_at"])
        results = []
        for name in self._registry:
            for version, info in self._registry[name].items():
                if version != "latest":
                    results.append(info.to_dict())
        return results
