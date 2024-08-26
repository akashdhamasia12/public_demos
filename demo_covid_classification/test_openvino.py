import torch
import torch.nn as nn
from torchvision import transforms as transforms
import random
from torchvision.models import vgg19_bn, vgg16, resnet18, resnet50, densenet121
from dataset import test_loader
from utils import compute_metrics
from SimpleCNN import SimpleCNN
import matplotlib.pyplot as plt 
from openvino.runtime import Core
from pathlib import Path
from IPython.display import Markdown, display
import os

random.seed(100)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

models_list = ["SimpleCNN_test"]
# models_list = ["SimpleCNN", "vgg19_bn", "vgg16", "ResNet18", "ResNet50", "Dense121", "efficientNet-b0"]
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

    model_path = Path("COVID-CT/storage/best_model_" + modelname).with_suffix(".pth")
    onnx_path = model_path.with_suffix(".onnx")
    ir_path = model_path.with_suffix(".xml")
    #Convert PyTorch model to ONNX

    if not onnx_path.exists():
        dummy_input = torch.randn(1, 3, 224, 224)

        # For the Fastseg model, setting do_constant_folding to False is required
        # for PyTorch>1.5.1
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=11,
            do_constant_folding=False,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")


    # Construct the command for Model Optimizer.
    mo_command = f"""mo
                    --input_model "{onnx_path}"
                    --input_shape "[1,3, {224}, {224}]"
                    --mean_values="[123.675, 116.28 , 103.53]"
                    --scale_values="[58.395, 57.12 , 57.375]"
                    --data_type FP16
                    --output_dir "{model_path.parent}"
                    """
    mo_command = " ".join(mo_command.split())
    print("Model Optimizer command to convert the ONNX model to OpenVINO:")
    # display(Markdown(f"`{mo_command}`"))
    print(mo_command)

    if not ir_path.exists():
        print("Exporting ONNX model to IR... This may take a few minutes.")        
        os.system(mo_command)
        # mo_result = %sx $mo_command
        # print("\n".join(mo_result))
    else:
        print(f"IR model {ir_path} already exists.")
    # metrics_dict = compute_metrics(model, test_loader, plot_roc_curve = False)

    # accuracy_list.append(metrics_dict['Accuracy'])

    # print('------------------- Test Performance --------------------------------------')
    # print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
    # print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
    # print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
    # print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
    # print("------------------------------------------------------------------------------")






