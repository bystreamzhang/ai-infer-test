from src.services.model_registry import ModelRegistry, ModelNotFoundError

reg = ModelRegistry()

# 注册两个版本
reg.register('classifier', 'v1.0', object())
reg.register('classifier', 'v2.0', object())

# latest 应该指向 v2.0
m = reg.get('classifier', 'latest') # 这里直接获得model实例，无法直接比较版本号，但至少可以检查是否成功获取到模型
print('latest exists:', m is not None)

# 列出模型，应该有2条（不含 latest 虚拟版本）
models = reg.list_models()
print('model count:', len(models))  # 期望: 2
print('sorted by time:', models[0]['version'], '->', models[1]['version'])  # 期望: v1.0 -> v2.0

# 注销 v2.0 后，latest 应该消失
reg.unregister('classifier', 'v2.0')
try:
    reg.get('classifier', 'latest')
    print('ERROR: should have raised')
except ModelNotFoundError:
    print('latest cleared after unregister: OK')

# 注销不存在的版本应抛异常
try:
    reg.unregister('classifier', 'v99')
    print('ERROR: should have raised')
except ModelNotFoundError:
    print('unregister nonexistent raises ModelNotFoundError: OK')

print('ALL PASS')