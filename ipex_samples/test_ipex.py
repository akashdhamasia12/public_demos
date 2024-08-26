import torch
import torchvision
import time
    
data = torch.rand(1, 3, 224, 224)
model = torchvision.models.resnet50(pretrained=False)
model.eval()

# #Run with IPEX
# import intel_extension_for_pytorch as ipex
# # model = ipex.optimize(model)
# model = ipex.optimize(model, dtype=torch.bfloat16)

import intel_extension_for_pytorch as ipex
model = torch.compile(model, backend="ipex")

#Warmup
for _ in range(10):
    with torch.no_grad():
        #with torch.cpu.amp.autocast():
        y = model(data)

dur = []

for _ in range(10):
    with torch.no_grad():
        #with torch.cpu.amp.autocast():
        with torch.autograd.profiler_legacy.profile() as prof:
            tic = time.time()
            y = model(data)
            d = time.time() - tic
            dur.append(d)

# duration = (time.time() - tic)/10
duration = sum(dur) / len(dur)


print(prof.key_averages().table(sort_by='self_cpu_time_total'))
# print(prof.table(sort_by='self_cpu_time_total'))
print("Inference time: {}ms".format((duration)*1000))


