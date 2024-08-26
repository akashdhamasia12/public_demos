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

ipex.nn.utils._model_convert.replace_dropout_with_identity(model)
dumpy_tensor = torch.ones((1, 3, 2627, 3623), dtype=torch.float)
jit_inputs = (dumpy_tensor)

from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8), weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
#qconfig = ipex.quantization.default_static_qconfig
prepared_model = prepare(model, qconfig, example_inputs=jit_inputs, inplace=False)

#model = ipex.optimize(model, dtype=torch.bfloat16)

import time

with torch.no_grad():
    with torch.cpu.amp.autocast():
        converted_model = convert(prepared_model)
        traced_model = torch.jit.trace(converted_model, jit_inputs, strict=False)
        traced_model = torch.jit.freeze(traced_model)
# model.save('quantized_jit_model_bf16-int8.pt')

        # Warmup
        for _ in range(10):
            y = traced_model(x)

        # tic = time.time()
        dur = []

        for _ in range(10):
            with torch.autograd.profiler_legacy.profile() as prof:
                tic = time.time()
                y = traced_model(x)
                d = time.time() - tic
                dur.append(d)

        # print("Inference time: {}s".format((time.time() - tic)/10))
        # duration = (time.time() - tic)/10
        duration = sum(dur) / len(dur)

        print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        print("Inference time: {}s".format((duration)))
        print("Throughput: {}fps".format(1/duration))
