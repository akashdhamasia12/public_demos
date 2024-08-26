# ignore all warning messages
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

import config_devcloud as config
from resnet50_dropout_devcloud import resnet50
from dataset_devcloud import test_loader
from plot_trajectory_devcloud import trajectories_plot

sequence_length = config.SEQ_LENGTH
past_trajectory = config.HISTORY
history_frames = past_trajectory*2 + 3
total_maneuvers = ["none", "straight", "right", "left"]

# Model Creation
def build_model() -> torch.nn.Module:
    # load pre-trained Conv2D model
    # '''
    model = resnet50(pretrained=True, p=config.dropout_prob)

    # change input channels number to match the rasterizer's output
    num_in_channels = 3 + (2*past_trajectory)

    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states

    if config.future_prediction > 0:
        num_targets = 2 * config.future_prediction
    else:
        num_targets = 2 * (sequence_length - past_trajectory)

    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model
    

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)


model_path = "best_epoch_MSE_.pth"
# model_path = config.model_path

model = build_model().to(config.DEVICE)
model = WrappedModel(model) #the model was trained on gpu with dataparallel,, need to add this to run on cpu.

# # load the model checkpoint
# print('Loading checkpoint')
# checkpoint = torch.load(config.model_path, map_location=config.DEVICE)
checkpoint = torch.load(model_path, map_location=config.DEVICE)

model_epoch = checkpoint['epoch']
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])#, strict=False) # Error(s) in loading state_dict for ResNet: -> added strict-False
# print('Loaded checkpoint at epoch', model_epoch)

model.eval()

def calculate_ade(outputs, targets):

    displacement = np.linalg.norm(outputs - targets, axis=1) 
    ade = np.mean(displacement)   
    return ade

def calculate_fde(outputs, targets):

    fde = np.linalg.norm(outputs[outputs.shape[0]-1] - targets[outputs.shape[0]-1])
    return fde


def evaluate(model, dataloader, ade_list, fde_list, plot_figures, mixedprecision):

    test_time_list = []

    with torch.no_grad():

        for i, data in enumerate(dataloader):

            # data = data.to(memory_format=torch.channels_last)

            image, keypoints, availability, seq_id, image_agent, centroid_current, history_traj, history_traj_availability = data['image'].to(config.DEVICE), torch.squeeze(data['keypoints'].to(config.DEVICE)), torch.squeeze(data['availability'].to(config.DEVICE)), torch.squeeze(data['seq_id'].to(config.DEVICE)), torch.squeeze(data['current_agent_i'].to(config.DEVICE)), torch.squeeze(data['centroid_current'].to(config.DEVICE)), torch.squeeze(data['history_traj'].to(config.DEVICE)), torch.squeeze(data['history_traj_availability'].to(config.DEVICE))

            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            keypoints = keypoints.detach().cpu().numpy()
            keypoints = ((keypoints + 1)/2)*int(config.IMAGE_SIZE)

            availability = availability.view(availability.size(0), -1)
            availability = availability.detach().cpu().numpy()

            history_traj = history_traj.view(history_traj.size(0), -1)
            history_traj_availability = history_traj_availability.view(history_traj_availability.size(0), -1)
            image_agent = image_agent.detach().cpu().numpy()
            centroid_current = centroid_current.detach().cpu().numpy()
            history_traj = history_traj.detach().cpu().numpy()
            history_traj_availability = history_traj_availability.detach().cpu().numpy()

            # outputs, _ = model.forward(image)
            t1 = time.time()

            if mixedprecision:
                with torch.cpu.amp.autocast():
                    outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping
            else:
                outputs, _ = model(image) #outputs=model(image).reshape(keypoints.shape)-->since we flattened the keypoints, no need for reshaping

            test_time = time.time() - t1

            test_time_list.append(test_time)

            outputs = outputs.view(keypoints.shape[0], -1)
            outputs = outputs.detach().cpu().numpy()
            outputs = ((outputs + 1)/2)*int(config.IMAGE_SIZE)
            outputs = outputs[np.where(availability == 1)]
    
            image = image.detach().cpu().numpy()

            keypoints = keypoints[np.where(availability == 1)]
            keypoints = keypoints.reshape(-1,2)

            history_traj = history_traj[np.where(history_traj_availability == 1)]
            history_traj = history_traj.reshape(-1,2)

            # outputs_mean = np.asarray(outputs_list[0])
            outputs = outputs.reshape(-1,2)
            
            ade = calculate_ade(outputs, keypoints) * config.IMAGE_FACTOR
            fde = calculate_fde(outputs, keypoints) * config.IMAGE_FACTOR

            ade_list.append(ade)
            fde_list.append(fde)

            # print("seq_id, ADE(px), FDE(px) = ", seq_id.item(), ade, fde)
            # f.write(str(seq_id.item())+","+str(ade)+","+str(fde)+"\n")

            trajectories_plot(image, outputs, keypoints, seq_id.item(), image_agent, ade, fde, history_traj, plot_figures)

        return ade_list, fde_list, test_time_list


#Run with IPEX
import intel_extension_for_pytorch as ipex
model_ipex = ipex.optimize(model)

ade_list = []
fde_list = []

f=open(f"{config.plots}/ade-fde_queue_ipex.txt","w+")

plot_figures = False
mixedprecision = False

# validatioon function
ade_list, fde_list, test_time_list_ipex = evaluate(model_ipex, test_loader, ade_list, fde_list, plot_figures, mixedprecision)

# print("total_ADE(in pixels) =", sum(ade_list)/len(ade_list))
# print("total_FDE(in pixels) =", sum(fde_list)/len(fde_list))
# print("total_model_time(s) =", sum(test_time_list_ipex)/len(test_time_list_ipex))

total_ADE_ipex = sum(ade_list)/len(ade_list)

f.write(str(sum(test_time_list_ipex)/len(test_time_list_ipex)) +"\n")
f.write(str(total_ADE_ipex))
f.close()

print("code ran successfully")