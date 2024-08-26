#!/usr/bin/env python
# encoding: utf-8
import time
import torch
import torchvision.models

data=torch.rand(1, 3, 224, 224)
net = torchvision.models.resnet50(pretrained=True)
net.eval()
layer = torch.nn.Softmax()

def get_softmax_prob(pred):
    return layer(pred)

with torch.autograd.profiler.emit_itt():
  with torch.no_grad():

    for i in range(10):
      #torch.profiler.itt.range_push('warmup_{}'.format(i))
      with torch.profiler.itt.range('warmup_{}'.format(i)):
        pred = net(data)

      with torch.profiler.itt.range('warmup_softmax_{}'.format(i)):
        class_prob = get_softmax_prob(pred)

      #torch.profiler.itt.range_pop()
    t0 = time.time()
    for i in range(10):
      #torch.profiler.itt.range_push('benchmark_{}'.format(i))
      with torch.profiler.itt.range('benchmark_{}'.format(i)):
        pred = net(data)

      with torch.profiler.itt.range('benchmark_softmax_{}'.format(i)):
        class_prob = get_softmax_prob(pred)
      #torch.profiler.itt.range_pop()
    t1 = time.time()
    print('time: {:.5f} ms/f'.format((t1 - t0) / 10 * 1000))