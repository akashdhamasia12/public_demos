import numpy as np 
# import matplotlib.pyplot as plt

#PyTorch - Importing the Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms

epochs = 1 

#PyTorch - Getting and Splitting the Dataset
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset_pytorch = torchvision.datasets.FashionMNIST(root='./data/',
                                             train=True, 
                                             transform=transforms,
                                             download=True)
test_dataset_pytorch = torchvision.datasets.FashionMNIST(root='.data/',
                                             train=False, 
                                             transform=transforms,
                                             download=True)


#PyTorch - Loading the Data
# def imshowPytorch(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset_pytorch,
                                           batch_size=1, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset_pytorch,
                                           batch_size=1, 
                                           shuffle=False)
                                           
# data_iter = iter(train_loader)
# images, label = data_iter.next()
# imshowPytorch(torchvision.utils.make_grid(images[0]))
# print(label[0])

#PyTorch - Building the Model
class NeuralNet(nn.Module):
    def __init__(self, num_of_class):
        super(NeuralNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(-1, 16*5*5)
        x = self.fc_model(x)
        x = self.classifier(x)
        return x

#PyTorch - Visualizing the Model
modelpy = NeuralNet(10)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(modelpy.parameters())

print(modelpy)

# #PyTorch - Training the Model
# for e in range(epochs):
#     # define the loss value after the epoch
#     losss = 0.0
#     number_of_sub_epoch = 0
    
#     # loop for every training batch (one epoch)
#     for images, labels in train_loader:
#         #create the output from the network
#         out = modelpy(images)
#         # count the loss function
#         loss = criterion(out, labels)
#         # in pytorch you have assign the zero for gradien in any sub epoch
#         optim.zero_grad()
#         # count the backpropagation
#         loss.backward()
#         # learning
#         optim.step()
#         # add new value to the main loss
#         losss += loss.item()
#         number_of_sub_epoch += 1
#     print("Epoch {}: Loss: {}".format(e, losss / number_of_sub_epoch))


#PyTorch - Comparing the Results
correct = 0
total = 0
modelpy.eval()
for images, labels in test_loader:
    for i in range(10):
        outputs = modelpy(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    break
print('Test Accuracy of the model on the {} test images: {}% with PyTorch'.format(total, 100 * correct // total))