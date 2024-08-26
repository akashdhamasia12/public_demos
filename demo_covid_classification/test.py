import torch
import torch.nn as nn
from torchvision import transforms as transforms
import random
from torchvision.models import vgg19_bn, vgg16, resnet18, resnet50, densenet121
from dataset import test_loader
from utils import compute_metrics
from SimpleCNN import SimpleCNN
import matplotlib.pyplot as plt 

random.seed(100)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

# models_list = ["SimpleCNN_test"]
models_list = ["SimpleCNN", "vgg19_bn", "vgg16", "ResNet18", "ResNet50", "Dense121", "efficientNet-b0"]
accuracy_list = []

for modelname in models_list:

    if modelname == 'SimpleCNN':
        model = SimpleCNN()
    elif modelname == 'SimpleCNN_test':
        model = SimpleCNN()
    elif modelname == 'vgg19_bn':
        model = vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 2)
    elif modelname == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 2)
    elif modelname == 'ResNet18':
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=2)
    elif modelname == 'ResNet50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=2)
    elif modelname == 'Dense121':
        model = densenet121(pretrained=True)
        model.classifier = nn.Linear(1024,2)
    elif modelname == 'efficientNet-b0':
        ### efficientNet
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

    print(modelname)
    model.to(device)

    model = torch.load("COVID-CT/storage/best_model_" + modelname + "_.pkl")

    metrics_dict = compute_metrics(model, test_loader, plot_roc_curve = False)

    accuracy_list.append(metrics_dict['Accuracy'])

    print('------------------- Test Performance --------------------------------------')
    print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
    print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
    print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
    print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
    print("------------------------------------------------------------------------------")


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(models_list, accuracy_list, color ='maroon', width = 0.4)
 
plt.xlabel("models")
plt.ylabel("Accuracy")
plt.title("Covid Classification with various models.")
plt.show()

fig.savefig("Covid_classification_models.png")

