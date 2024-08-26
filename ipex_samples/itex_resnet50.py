# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))

import tensorflow_hub as hub
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import urllib
import os
import sys
import time


def main():
    module = hub.KerasLayer("https://tfhub.dev/google/supcon/resnet_v1_50/imagenet/classification/1")
    input = np.random.rand(1, 224, 224, 3)
    x = preprocess_input(input, mode='tf')

    #warmup
    durs = []
    for i in range(10):
        #itt.task_begin(domain, 'warmup_{}'.format(i))
        t0 = time.time()
        module(x)
        # torch.xpu.synchronize()
        t1 = time.time()
        durs.append(t1-t0)
        print(t1-t0)
        #itt.task_end(domain)
    print('warmup dur: {}ms'.format(sum(durs) / len(durs) * 1000))

    durs = []
    for i in range(10):
        #itt.task_begin(domain, 'benchmark_{}'.format(i))
        t0 = time.time()
        module(x)
        # torch.xpu.synchronize()
        t1 = time.time()
        durs.append(t1-t0)
        #itt.task_end(domain)
    #print('dur: {}ms'.format(durs[int(len(durs)/2)] * 1000))
    print('dur: {}ms'.format(sum(durs) / len(durs) * 1000))


if __name__ == "__main__":
    main()