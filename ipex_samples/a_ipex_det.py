import io
import torch
import torch.onnx
from ocr.source.detection.models.model import DBModel
#from ocr.source.detection.models.modules import *
import intel_extension_for_pytorch as ipex

device = 'cpu'
x = torch.rand(1, 3, 2627, 3623).to(device)

pthfile = 'models/Detection_1.0.pth'

loaded_model = torch.load(pthfile, map_location='cpu')

config = loaded_model['config']
conf_dict = {}
conf_dict['backbone'] = config['arch']['args']['backbone']
conf_dict['pretrained'] = config['arch']['args']['pretrained']
conf_dict['segmentation_body'] = config['arch']['args']['segmentation_body']
conf_dict['segmentation_head'] = config['arch']['args']['segmentation_head']

model = DBModel(model_config=conf_dict).to(device)
model.eval()

# model = model.to(memory_format=torch.channels_last)
# x = x.to(memory_format=torch.channels_last)
model = ipex.optimize(model)

import time

#Warmup
for _ in range(10):
    with torch.no_grad():
        #with torch.cpu.amp.autocast():
        y = model(x)

dur = []

for _ in range(10):
    with torch.no_grad():
        #with torch.cpu.amp.autocast():
        with torch.autograd.profiler_legacy.profile() as prof:
            tic = time.time()
            y = model(x)
            d = time.time() - tic
            dur.append(d)

# duration = (time.time() - tic)/10
duration = sum(dur) / len(dur)


print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.table(sort_by='self_cpu_time_total'))
print("Inference time: {}s".format((duration)))
print("Throughput: {}fps".format(1/duration))


