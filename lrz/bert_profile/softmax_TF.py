#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from keras.applications.resnet import ResNet50
import time

data = tf.random.normal([1,224,224,3])

model = ResNet50(weights='imagenet')
layer = tf.keras.layers.Softmax()

def get_softmax_prob(pred):
    return layer(pred)

#warm up
for i in range(10):
  preds = model.predict(data)
  class_prob = get_softmax_prob(preds)

t0 = time.time()
for i in range(10):
  preds = model.predict(data)
  class_prob = get_softmax_prob(preds)
t1 = time.time()
print('time: {:.5f} ms/f'.format((t1 - t0) / 10 * 1000))