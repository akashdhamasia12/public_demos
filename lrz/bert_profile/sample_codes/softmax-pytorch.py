import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T


model = torch.nn.Sequential(
    torch.nn.Linear(3,3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(3,3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(3,3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Softmax(dim=1)
)

print(model)


data = pd.read_csv("data.csv")

data_x = np.array(data[["plastic","paper","glass"]], dtype=np.float32)
data_y = np.array(data[["student","worker","elder"]], dtype=np.float32)
x_train = torch.from_numpy(data_x)
y_train = torch.from_numpy(data_y)
num_epoch = 1000

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/softmax-pytorch'),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        use_cuda=False)

prof.start()

for epoch in range(num_epoch):
    input = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(input)
    loss = loss_function(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    prof.step()

    # show
    print('Epoch[{}/{}], loss: {:.6f}'
          .format(epoch + 1, num_epoch, loss.data.item()))

# predicting
print(model(torch.tensor([[500, 500, 500]], dtype=torch.float32)))