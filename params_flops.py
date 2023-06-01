# from DCNND import DCNND
# from DACNN import DACNN
import torch
# from torchvision.models import resnet50
# from models_abl8 import DsCNN
from models_t2 import DsCNN

model=DsCNN()
# model = DACNN()
print(model)
# model = DCNND()
# model1 = resnet50()
# model_name = DACNN
"""通过torchstat.stat 可以查看网络模型的参数量和计算复杂度FLOPs"""
from thop import profile
# input = torch.randn(1,3,224,224)
input = torch.randn(1,1,140,140)
flops, params = profile(model, inputs=(input,))
print('the params is {} M, the flops is {} G. '.format(round(params / (10 ** 6), 2),
                                                             round(flops / (10 ** 9),
                                                                   2)))  # 4111514624.0 25557032.0 res50
