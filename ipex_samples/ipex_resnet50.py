#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision
import intel_extension_for_pytorch as ipex
#from torch.jit._recursive import wrap_cpp_module
#import itt

def main():
#    device = 'xpu:0'
    device = 'xpu'
    net = torchvision.models.resnet50()
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    input = torch.rand(1, 3, 224, 224)
    net = net.to(device)
    input = input.to(device)
    print("akash1")
    #domain = itt.domain_create("domain")
    with torch.no_grad():
        net.eval()
        net = ipex.optimize(net, dtype=torch.float32)
 #       net = torch.jit.script(net)
 #       net = wrap_cpp_module(torch._C._jit_pass_fold_convbn(net._c))
        #net = torch.jit.trace(net, input)
        #net = torch.jit.freeze(net)
        print("akash2")

        #warmup
        durs = []
        for i in range(10):
            #itt.task_begin(domain, 'warmup_{}'.format(i))
            t0 = time.time()
            net(input)
            torch.xpu.synchronize()
            t1 = time.time()
            durs.append(t1-t0)
            print(t1-t0)
            #itt.task_end(domain)
            print("akash3")
        print('warmup dur: {}ms'.format(sum(durs) / len(durs) * 1000))

        durs = []
        for i in range(10):
            #itt.task_begin(domain, 'benchmark_{}'.format(i))
            t0 = time.time()
            net(input)
            torch.xpu.synchronize()
            t1 = time.time()
            durs.append(t1-t0)
            print("kash4")
            #itt.task_end(domain)
        #print('dur: {}ms'.format(durs[int(len(durs)/2)] * 1000))
        print('dur: {}ms'.format(sum(durs) / len(durs) * 1000))

if __name__ == '__main__':
    main()
