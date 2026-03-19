from src.models.text_classifier import TextClassifier
clf = TextClassifier()

# 单条预测
r = clf.predict('NBA总决赛湖人队夺冠')
print('单条:', r)

# 批量预测
results = clf.predict_batch([
    '股市大涨科技股领涨',
    '新款手机发布搭载AI芯片',
    '明星新电影票房破十亿',
])
for r in results:
    print('批量:', r)