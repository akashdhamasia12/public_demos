#!/usr/bin/env python
# encoding: utf-8
import argparse
import time
import torch
# import resnet
import torchvision.models

parser = argparse.ArgumentParser()
parser.add_argument('--ipex', default=False, action="store_true")
parser.add_argument('--torchscript', default=False, action="store_true")
args = parser.parse_args()
data=torch.rand(1, 3, 224, 224)
net = torchvision.models.resnet50(pretrained=False)
net.eval()
if args.ipex:
  net = net.to(memory_format=torch.channels_last)
  data = data.to(memory_format=torch.channels_last)
  # import intel_extension_for_pytorch as ipex
  # net = ipex.optimize(net)
with torch.autograd.profiler.emit_itt():
  with torch.no_grad():
    if args.torchscript:
      net = torch.jit.trace(net, data)
      net = torch.jit.freeze(net)
      net(data)
    for i in range(10):
      #torch.profiler.itt.range_push('warmup_{}'.format(i))
      with torch.profiler.itt.range('warmup_{}'.format(i)):
        net(data)
      #torch.profiler.itt.range_pop()
    t0 = time.time()
    for i in range(10):
      #torch.profiler.itt.range_push('benchmark_{}'.format(i))
      with torch.profiler.itt.range('benchmark_{}'.format(i)):
        net(data)
      #torch.profiler.itt.range_pop()
    t1 = time.time()
    print('time: {:.5f} ms/f'.format((t1 - t0) / 10 * 1000))