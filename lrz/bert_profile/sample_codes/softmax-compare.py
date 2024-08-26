
import torch
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
# import intel

input = torch.randn(2, 3)

m = torch.nn.Softmax(dim=1)

for i in range(0,100):
    softmax_pytorch = m(input)

start_time = time.time()
softmax_pytorch = m(input)
softmax_pytorch_time = time.time() - start_time
print("---softmax_pytorch %s seconds ---" % (softmax_pytorch_time))
print("softmax_pytorch", softmax_pytorch)

for i in range(0,100):
    softmax_tf = tf.nn.softmax(input)

start_time = time.time()
softmax_tf = tf.nn.softmax(input)
softmax_tf_time = time.time() - start_time
print("---softmax_tf %s seconds ---" % (softmax_tf_time))
print("softmax_tf", softmax_tf)

speed_up = ((softmax_tf_time - softmax_pytorch_time)/softmax_pytorch_time)*100

   
names = ["tensorflow_" + tf.__version__, "pytorch" + torch.__version__]
values = [softmax_tf_time, softmax_pytorch_time]

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(names, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("framework")
plt.ylabel("time(s)")
plt.title("Softmax comparison Pytorch & Tensorflow Speed-up = " + str(round(speed_up,2)) + "%")
# plt.show()
plt.savefig('foo.png')