import torch
import os

#Config for Inference
model_path = "/home/sdp/akashd/best_epoch_MSE_.pth"
future_prediction = 25 #25 #number of frames to predict for future. else -1 for default
DATASET = "LYFT" #"LYFT"
IMAGE_SIZE = 224 #default: LYFT: 224 NGSIM: 300
DEVICE = torch.device('cpu') 

seed = 42 
dropout_prob = 0.2

if DATASET == "LYFT":
    DATASET_PATH = "lyft_sample_dataset_90" #new: ego vehicle at center of the map always now.
    IMAGE_FACTOR = 224 / IMAGE_SIZE
    SEQ_LENGTH = 41 #(10+1=history + 30 future)
    HISTORY = 11
    TEST_SPLIT = 0.1 #increased validation dataset to stop volatililty of validation loss
    MANEUVER_PRESENT = True #True 

model_name = "cnn_dn_" + str(future_prediction)
OUTPUT_PATH = DATASET_PATH + '/outputs_' + model_name + '_' + str(seed) 

plots = OUTPUT_PATH + "/plots"

if not os.path.exists(plots):
    os.makedirs(plots)

noise = False #random normal distribution to the image.
noise_freq = 8
