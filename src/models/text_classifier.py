import time
import random
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.utils.logger import get_logger

logger = get_logger("text_classifier")

# 内置训练数据：4个类别，每类20条中文短文本
_TRAIN_DATA: list[tuple[str, str]] = [
    # 体育
    ("中国队赢得世界杯小组赛首胜", "sports"),
    ("NBA总决赛湖人队夺冠", "sports"),
    ("奥运会游泳比赛打破世界纪录", "sports"),
    ("足球运动员签约新球队转会费创纪录", "sports"),
    ("羽毛球公开赛中国选手包揽金银牌", "sports"),
    ("马拉松比赛吸引两万名跑者参与", "sports"),
    ("乒乓球世锦赛中国队蝉联冠军", "sports"),
    ("篮球联赛季后赛激战正酣", "sports"),
    ("排球世界联赛中国女排大获全胜", "sports"),
    ("网球大满贯决赛令人窒息的逆转", "sports"),
    ("冬奥会短道速滑夺金时刻", "sports"),
    ("田径世锦赛百米赛跑新纪录", "sports"),
    ("高尔夫球赛冠军赢得巨额奖金", "sports"),
    ("赛车手在F1比赛中完成精彩超越", "sports"),
    ("拳击世界冠军卫冕成功", "sports"),
    ("体操世界杯中国队斩获多枚金牌", "sports"),
    ("帆船比赛遭遇大风浪选手坚持完赛", "sports"),
    ("射击比赛打破奥运会纪录", "sports"),
    ("跆拳道赛场精彩对决", "sports"),
    ("举重运动员挑战极限刷新全国纪录", "sports"),
    # 科技
    ("人工智能大模型发布引发行业震动", "tech"),
    ("量子计算机突破传统计算极限", "tech"),
    ("芯片制造工艺进入埃米时代", "tech"),
    ("自动驾驶汽车获批上路测试", "tech"),
    ("新型电池续航里程突破一千公里", "tech"),
    ("卫星互联网覆盖全球偏远地区", "tech"),
    ("机器人完成高难度外科手术", "tech"),
    ("虚拟现实头显实现真正沉浸体验", "tech"),
    ("区块链技术在供应链中落地应用", "tech"),
    ("基因编辑技术治愈遗传性疾病", "tech"),
    ("超导材料在常温下实现突破", "tech"),
    ("5G网络切片技术赋能工业互联网", "tech"),
    ("云计算平台推出新一代AI服务", "tech"),
    ("开源大语言模型超越商业产品", "tech"),
    ("脑机接口实现思维控制机械臂", "tech"),
    ("折叠屏手机迎来新一代产品升级", "tech"),
    ("无人机集群完成复杂编队表演", "tech"),
    ("新型激光雷达成本大幅降低", "tech"),
    ("边缘计算芯片实现低功耗推理", "tech"),
    ("操作系统内核重大安全漏洞修复", "tech"),
    # 娱乐
    ("国产电影票房突破百亿创历史", "entertainment"),
    ("流行歌手新专辑首日播放量破纪录", "entertainment"),
    ("综艺节目收视率连续夺冠", "entertainment"),
    ("电视剧大结局引发全网热议", "entertainment"),
    ("明星婚礼成为年度最受关注事件", "entertainment"),
    ("动漫电影斩获国际大奖", "entertainment"),
    ("游戏直播主播粉丝突破千万", "entertainment"),
    ("短视频平台推出全新创作激励计划", "entertainment"),
    ("音乐节吸引数十万乐迷齐聚", "entertainment"),
    ("话剧演出场场爆满一票难求", "entertainment"),
    ("网络小说改编剧集引爆话题", "entertainment"),
    ("喜剧演员巡回演出圆满落幕", "entertainment"),
    ("选秀节目新星引发粉丝追捧", "entertainment"),
    ("电影节颁奖典礼惊喜连连", "entertainment"),
    ("偶像团体出道演唱会座无虚席", "entertainment"),
    ("脱口秀节目金句频出刷爆朋友圈", "entertainment"),
    ("古装剧服化道获专业人士高度赞扬", "entertainment"),
    ("纪录片深度揭示行业幕后故事", "entertainment"),
    ("跨年晚会明星阵容豪华观众期待", "entertainment"),
    ("电影续集票房超越前作", "entertainment"),
    # 财经
    ("A股市场成交量创年内新高", "finance"),
    ("人民币兑美元汇率小幅走强", "finance"),
    ("央行宣布降准释放长期流动性", "finance"),
    ("科技股领涨带动大盘全面回升", "finance"),
    ("新能源板块集体爆发涨停潮", "finance"),
    ("基金公司发行新产品首日售罄", "finance"),
    ("房地产市场政策调整影响深远", "finance"),
    ("黄金价格创历史新高避险需求旺盛", "finance"),
    ("上市公司发布靓丽年报股价大涨", "finance"),
    ("外资持续流入A股市场信心增强", "finance"),
    ("债券市场收益率持续下行", "finance"),
    ("创业板注册制改革提升市场活力", "finance"),
    ("大宗商品价格波动影响通胀预期", "finance"),
    ("银行业净息差收窄面临转型压力", "finance"),
    ("保险公司加速布局养老金融赛道", "finance"),
    ("数字人民币试点城市扩大范围", "finance"),
    ("PE机构加大对硬科技领域投资", "finance"),
    ("IPO审核趋严退市机制持续完善", "finance"),
    ("跨境电商推动外贸新增长极", "finance"),
    ("碳交易市场扩容覆盖更多行业", "finance"),
]


class TextClassifier:
    """基于TF-IDF + 朴素贝叶斯的文本分类器。

    使用内置的中文短文本数据集训练，支持体育/科技/娱乐/财经四个类别。
    """

    def __init__(self) -> None:
        """初始化分类器：构建pipeline并用内置数据训练。"""
        logger.info("initializing TextClassifier")

        # TODO: 构建 sklearn Pipeline
        # 两个步骤：
        #   1. ("tfidf", TfidfVectorizer()) — 把文本变成TF-IDF向量
        #      提示：加参数 analyzer="char_wb", ngram_range=(2, 3) 让中文效果更好
        #   2. ("clf", MultinomialNB()) — 朴素贝叶斯分类器
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))),
            ("clf", MultinomialNB())
        ])

        # 拆分训练数据
        texts = [text for text, _ in _TRAIN_DATA]
        labels = [label for _, label in _TRAIN_DATA]

        # TODO: 调用 self.pipeline.fit(texts, labels) 训练模型
        self.pipeline.fit(texts, labels)

        logger.info("TextClassifier ready", num_classes=len(set(labels)), num_samples=len(texts))

    def predict(self, text: str) -> dict[str, Any]:
        """对单条文本进行分类预测。

        Args:
            text: 待分类的文本字符串。

        Returns:
            包含以下字段的字典：
                label: 预测类别（"sports"/"tech"/"entertainment"/"finance"）
                confidence: 最大类别的概率值，范围 [0, 1]
                latency_ms: 推理耗时（毫秒）
        """
        start = time.perf_counter()

        # 模拟真实推理延迟
        time.sleep(random.uniform(0.01, 0.05))

        # TODO: 用 predict_proba 获取概率分布
        # self.pipeline.predict_proba([text]) 返回 shape (1, n_classes) 的 numpy array
        # 注意：输入必须是列表，不能直接传字符串
        probabilities: np.ndarray = self.pipeline.predict_proba([text])  # shape: (1, n_classes)

        # TODO: 找到最大概率对应的类别
        # 提示1：用 np.argmax(probabilities[0]) 找到最大概率的索引
        # 提示2：self.pipeline.classes_ 是类别标签数组，用上面的索引取出标签
        # 提示3：probabilities[0][index] 就是对应的置信度
        index = np.argmax(probabilities[0])
        label: str = self.pipeline.classes_[index].item()  # item() 转成纯 Python 字符串
        confidence: float = probabilities[0][index]

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug("predict done", label=label, confidence=round(confidence, 4))
        return {"label": label, "confidence": float(confidence), "latency_ms": latency_ms}

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """对多条文本批量预测。

        Args:
            texts: 待分类的文本列表。

        Returns:
            与输入顺序对应的预测结果列表，每个元素格式同 predict() 的返回值。
        """
        # TODO: 对 texts 中每条文本调用 self.predict(text)，收集结果
        # 提示：用列表推导式，一行搞定
        results: list[dict[str, Any]] = [self.predict(text) for text in texts]

        logger.info("batch predict done", count=len(results))
        return results
