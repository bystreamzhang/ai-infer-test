from src.models.text_generator import TextGenerator
g = TextGenerator()
result = g.generate('人工智能', max_length=30)
print('生成文本:', result['text'])
print('新增字符数:', result['tokens_generated'])
print('耗时:', round(result['latency_ms'], 2), 'ms')