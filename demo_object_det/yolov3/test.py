from __future__ import division
import time
import torch 
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image
import random 
import pickle as pkl
        
def get_test_input(input_dim, GPU):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if GPU:
        img_ = img_.cuda()
    return img_


def evaluate(model):
    
    images = "imgs" 
    batch_size = 1
    confidence = 0.5
    nms_thesh = 0.4
    GPU = False
    num_classes = 80
    classes = load_classes('data/coco.names')         
    inp_dim = 416

    
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
                    
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
        
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        


    i = 0
    write = False
    
    model(get_test_input(inp_dim, GPU), GPU)
    
    objs = {}
    
    test_time_list = []

    for batch in im_batches:        

        #Apply offsets to the result predictions
        with torch.no_grad():
            t1 = time.time()
            prediction = model(Variable(batch), GPU)
            test_time = time.time() - t1
            
            test_time_list.append(test_time)
        
        #NMS        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        
        # if type(prediction) == int:
        #     i += 1
        #     continue
                    

        prediction[:,0] += i*batch_size
          
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))
        

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted ".format(image.split("/")[-1]))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

            
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        

    colors = pkl.load(open("pallete", "rb"))
    

    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        # cv2.rectangle(img, c1, c2,color, 1)
        cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])),color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])),color, -1)
        cv2.putText(img, label, (int(c1[0]), int(c1[1] + t_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img
             
    images_det = list(map(lambda x: write(x, im_batches, orig_ims), output))
              
    time_test = sum(test_time_list)/len(test_time_list)
    
    return time_test, orig_ims

            
#Set up the neural network
print("Loading network.....")
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network successfully loaded")
model.net_info["height"] = "416"

#Set the model in evaluation mode
model.eval()
time_test, orig_ims = evaluate(model)

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)

time_test_ipex, orig_ims = evaluate(model)

print(time_test, time_test_ipex)
print("speedup= ", time_test/time_test_ipex)
