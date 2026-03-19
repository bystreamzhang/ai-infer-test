"""结构化日志配置模块。

使用 structlog 将所有日志以 JSON 格式输出，便于日志采集系统解析。
每条日志自动附带时间戳、日志级别和调用位置信息。
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """全局初始化 structlog，应在程序启动时调用一次。

    Args:
        log_level: 日志级别字符串，如 "DEBUG", "INFO", "WARNING"。
    """
    # TODO: 配置 structlog 的 processor chain
    # 需要按顺序添加以下 processor：
    #   1. structlog.stdlib.add_log_level        → 在 event_dict 中加入 "level" 字段
    #   2. structlog.stdlib.add_logger_name      → 加入 "logger" 字段（模块名）
    #   3. structlog.processors.TimeStamper(fmt="iso")  → 加入 ISO 格式时间戳
    #   4. structlog.processors.CallsiteParameterAdder(  → 加入调用位置
    #          [structlog.processors.CallsiteParameter.FILENAME,
    #           structlog.processors.CallsiteParameter.LINENO]
    #      )
    #   5. structlog.processors.StackInfoRenderer()      → 渲染异常栈
    #   6. structlog.processors.JSONRenderer()           → 最终渲染为 JSON 字符串
    processors = [
        # TODO: 在这里填入上面列出的 6 个 processor
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME,
             structlog.processors.CallsiteParameter.LINENO]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer()
    ]

    # TODO: 调用 structlog.configure(...)，传入以下参数：
    #   - processors=processors
    #   - wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[log_level])
    #   - context_class=dict
    #   - logger_factory=structlog.PrintLoggerFactory(file=sys.stdout)
    # 提示：structlog.configure 就是全局设置，类似于 logging.basicConfig
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[log_level]),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout)
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取一个绑定了模块名的 logger 实例。

    用法示例：
        log = get_logger(__name__)
        log.info("服务启动", port=8080)
        log.error("请求失败", status_code=500, path="/api/v1/classify")

    Args:
        name: 模块名，通常传入 __name__。

    Returns:
        绑定了 name 上下文的 structlog BoundLogger。
    """
    # TODO: 调用 structlog.get_logger(name) 并返回
    # 提示：structlog.get_logger() 类似于 logging.getLogger()，
    #       但返回的是 structlog 的 BoundLogger，支持链式 .bind()
    return structlog.get_logger().bind(logger=name)
