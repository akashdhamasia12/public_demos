import torch
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import copy

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}
    
model = torchvision.models.resnet18()
dnn_to_bnn(model, const_bnn_prior_parameters)


# model_1 = copy.deepcopy(model)

#Run with IPEX
import intel_extension_for_pytorch as ipex

model.eval()
# model_ipex = ipex.optimize(model)
model_ipex = model

print(model_ipex)

data = torch.rand(1, 3, 224, 224)

#warmup_
for _ in range(10):
    output = model_ipex(data)

import time
dur_list = []  
start = time.time()

for _ in range(10):
    with torch.no_grad():
        #with torch.cpu.amp.autocast():
        with torch.autograd.profiler_legacy.profile() as prof:
            output = model_ipex(data)
            dur = time.time() - start
            dur_list.append(dur)

print(prof.key_averages().table(sort_by='self_cpu_time_total'))
print('Inference took {:.2f} ms in average'.format(((sum(dur_list)/len(dur_list))*1000)))


