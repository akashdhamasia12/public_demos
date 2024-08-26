import torch
import torch.nn as nn
from torchvision import transforms as transforms
import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg19_bn, vgg16, resnet18, resnet50, densenet121
from dataset import train_loader, val_loader
from tqdm import tqdm
from utils import compute_metrics, EarlyStopping
from SimpleCNN import SimpleCNN

random.seed(100)

num_epochs = 60

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

# models_list = ["SimpleCNN", "vgg19_bn", "vgg16", "ResNet18", "ResNet50", "Dense121", "efficientNet-b0"]
models_list = ["SimpleCNN_test"]

for modelname in models_list:

    early_stopper = EarlyStopping(patience = 8)

    log_dir = "COVID-CT/logs_" + modelname 
    writer = SummaryWriter(log_dir)

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
    print(model)
    model.to(device)

    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

    best_model = model
    best_val_score = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        model.train()    
        train_loss = 0
        train_correct = 0
        
        for iter_num, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, target = data['img'].to(device), data['label'].to(device)     
            optimizer.zero_grad()

            # Compute the loss
            output = model(image)
            loss = criterion(output, target.long()) #/ 8
            
            # Log loss
            train_loss += loss.item()
            loss.backward()

            # Perform gradient udpate
            # if iter_num % 8 == 0:
            optimizer.step()
                

            # Calculate the number of correctly classified examples
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()
            
        
        # Compute and print the performance metrics
        metrics_dict = compute_metrics(model, val_loader)
        print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch,
                                                                                                    iter_num))
        print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
        print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
        print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
        print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
        print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
        print("------------------------------------------------------------------------------")
        
        # Save the model with best validation accuracy
        if metrics_dict['Accuracy'] > best_val_score:
            torch.save(model, "COVID-CT/storage/best_model_" + modelname + "_.pkl")
            best_val_score = metrics_dict['Accuracy']
        
        
        # print the metrics for training data for the epoch
        print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset)))
        
        # log the accuracy and losses in tensorboard
        writer.add_scalars( "Losses", {'Train loss': train_loss / len(train_loader), 'Validation_loss': metrics_dict["Validation Loss"]},
                                    epoch)
        writer.add_scalars( "Accuracies", {"Train Accuracy": 100.0 * train_correct / len(train_loader.dataset),
                                        "Valid Accuracy": 100.0 * metrics_dict["Accuracy"]}, epoch)

        # Add data to the EarlyStopper object
        early_stopper.add_data(model, metrics_dict['Validation Loss'], metrics_dict['Accuracy'])
        
        # If both accuracy and loss are not improving, stop the training
        if early_stopper.stop() == 1:
            break
        
        # if only loss is not improving, lower the learning rate
        if early_stopper.stop() == 3:
            for param_group in optimizer.param_groups:
                learning_rate *= 0.1
                param_group['lr'] = learning_rate
                print('Updating the learning rate to {}'.format(learning_rate))
                early_stopper.reset()
        