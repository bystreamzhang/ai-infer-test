import time
import random
from collections import defaultdict, deque
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("text_generator")

# 内置中文语料：用于构建 Markov Chain 转移表
_CORPUS: str = (
    "人工智能技术正在改变世界，深度学习模型不断突破性能极限。"
    "自然语言处理让机器理解人类语言，计算机视觉让机器看懂世界。"
    "大模型时代来临，语言模型的参数规模不断扩大，性能持续提升。"
    "量子计算机利用量子叠加态并行计算，有望解决传统计算机无法解决的问题。"
    "芯片制造工艺不断进步，晶体管密度持续提高，计算能力大幅增强。"
    "自动驾驶汽车通过传感器感知环境，利用算法做出驾驶决策。"
    "机器人技术结合人工智能，实现复杂任务的自主完成。"
    "云计算平台提供弹性计算资源，支撑大规模数据处理需求。"
    "区块链技术通过分布式账本保证数据不可篡改，实现去中心化信任。"
    "边缘计算将计算能力部署到数据产生的地方，降低延迟提高效率。"
    "数据是新时代的石油，数据分析帮助企业做出更好的决策。"
    "开源社区推动技术快速发展，开发者共同构建技术生态。"
    "网络安全威胁日益严峻，加密技术保护数据传输安全。"
    "5G网络高速低延迟的特性，为物联网和自动驾驶提供基础设施。"
    "虚拟现实和增强现实技术正在改变娱乐教育和工业应用场景。"
)


class TextGenerator:
    """基于 Markov Chain 的中文文本生成器。

    通过统计 N-gram 转移概率表，对给定 prompt 进行续写。
    使用内置中文语料构建转移表，支持可配置的 N-gram 阶数。
    """

    def __init__(self, order: int = 2) -> None:
        """初始化生成器：用内置语料构建转移概率表。

        Args:
            order: Markov Chain 阶数，即用前几个字预测下一个字。默认 2。
        """
        logger.info("initializing TextGenerator", order=order)

        self.order: int = order

        # 转移表：key 是长度为 order 的字符串（N-gram），
        # value 是该 N-gram 后面出现过的字符列表（列表允许重复，用于实现天然的加权采样）
        self.chain: dict[str, list[str]] = defaultdict(list)

        # 用于记录生成历史（看起来像调试用途）
        # self._history: list[str] = []
        self._history: deque[str] = deque(maxlen=100)  # 只保留最近 100 条记录

        self.build_chain(_CORPUS, order)

        logger.info("TextGenerator ready", chain_size=len(self.chain))

    def build_chain(self, corpus: str, order: int) -> None:
        """从语料构建 N-gram 转移表。

        遍历语料中的每个位置，把长度为 order 的子串作为 key，
        把紧随其后的字符追加到对应的 value 列表中。

        举例（order=2，语料="人工智能"）：
            "人工" → ["智"]
            "工智" → ["能"]

        Args:
            corpus: 用于构建转移表的文本语料。
            order: N-gram 阶数。
        """
        # TODO: 遍历 corpus，从索引 0 到 len(corpus) - order - 1
        # 在每个位置 i：
        #   - key = corpus[i : i + order]        （当前的 N-gram）
        #   - next_char = corpus[i + order]       （紧随其后的字符）
        #   - 把 next_char 追加到 self.chain[key] 中
        for i in range(0, len(corpus) - order):
            key = corpus[i : i + order]
            next_char = corpus[i + order]
            self.chain[key].append(next_char)

    def generate(self, prompt: str, max_length: int = 100) -> dict[str, Any]:
        """基于 prompt 续写文本。

        从 prompt 末尾取出 order 个字符作为初始 N-gram，
        反复查转移表并采样，逐字生成新内容，直到达到 max_length 或无路可走。

        Args:
            prompt: 生成的起始文本，长度必须 >= order。
            max_length: 最多生成的新字符数（不含 prompt 本身）。

        Returns:
            包含以下字段的字典：
                text: prompt + 生成的新内容
                tokens_generated: 实际生成的字符数
                latency_ms: 推理耗时（毫秒）
        """
        start = time.perf_counter()

        # 如果 prompt 太短，用语料开头补全到 order 个字符
        if len(prompt) < self.order:
            prompt = _CORPUS[: self.order]

        # 用列表拼接生成结果（比字符串反复拼接效率更高，类比 C++ 的 push_back）
        generated: list[str] = list(prompt)

        # TODO: 循环生成最多 max_length 个新字符
        # 每次迭代：
        #   1. 取 generated 末尾 self.order 个字符，拼成当前 N-gram：
        #      current_ngram = "".join(generated[-self.order :])
        #   2. 查 self.chain[current_ngram]，得到候选字符列表
        #      如果列表为空（走到了死路），break 终止
        #   3. 用 random.choice(candidates) 从候选列表中随机选一个字符
        #      （candidates 中重复的字符出现概率更高，天然实现了加权采样）
        #   4. 把选出的字符 append 到 generated
        for _ in range(max_length):
            current_ngram = "".join(generated[-self.order :])
            candidates = self.chain[current_ngram]
            if not candidates:
                break
            next_char = random.choice(candidates)
            generated.append(next_char)

        result_text = "".join(generated)
        tokens_generated = len(result_text) - len(prompt)

        # 记录生成历史，方便调试时回溯
        self._history.append(result_text)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "generate done",
            tokens_generated=tokens_generated,
            latency_ms=round(latency_ms, 2),
        )
        return {
            "text": result_text,
            "tokens_generated": tokens_generated,
            "latency_ms": latency_ms,
        }

    def predict(self, text: str) -> dict:
        return self.generate(text)