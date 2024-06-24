import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.optim import Adam
import math
import math
import torch
import torch.nn.functional as F
from torch import nn

class L1Attn(nn.Module):
    def __init__(self):
        super(L1Attn, self).__init__()
    def forward(self, q, k):
        bs, n_ctx, n_heads, width = q.shape#【128，28，1，10】
        scale = -1 / math.sqrt(width)

        qq = q.unsqueeze(1).expand([-1, n_ctx, -1, -1, -1])
        kk = k.unsqueeze(2).expand([-1, -1, n_ctx, -1, -1])
        ww = torch.abs(qq - kk) * scale
        attn = torch.sum(ww, -1)
        return attn

class L1_att_block(nn.Module):
    def __init__(self, input_size =28, d_model=28, n_head=1):
        super(L1_att_block, self).__init__()
        self.attn = L1Attn()
        self.n_heads = n_head
        self.k = nn.Linear(input_size, d_model)
        self.q = nn.Linear(input_size, d_model)
        self.v = nn.Linear(input_size, d_model)
        self.fc1 = nn.Linear(28, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
    def forward(self, x): 
        a, b, c, d = x.shape
        x = x.view(a, c, d) 
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        bs, n_ctx, width = x.shape
        attn_ch = width // self.n_heads 
        q = q.view(bs, n_ctx, self.n_heads, -1)
        k = k.view(bs, n_ctx, self.n_heads, -1)
        v = v.view(bs, n_ctx, self.n_heads, -1)
        weight = self.attn(q, k)
        k_indices = torch.arange(0, n_ctx)
        weight[:, k_indices, k_indices, :] = 1.0 
        L1_add_weight = torch.matmul(v,weight).reshape(bs,-1)
        out = self.fc1(L1_add_weight)
        out = self.relu(out)
        out = self.fc2(out)
        return out
import time
def adjust_learning_rate(epoch, optimizer):
    lr = 0.01
    if epoch > 180:
        lr = lr / 2
    elif epoch > 150:
        lr = lr / 2
    elif epoch > 120:
        lr = lr / 2
    elif epoch > 90:
        lr = lr / 2
    elif epoch > 60:
        lr = lr / 2
    elif epoch > 30:
        lr = lr / 2

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test(model, dataloader, device):
    model.eval().to(device)
    total_loss = 0
    total_correct = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input, tgt = batch[0].to(device), batch[1].to(device)
            result = model(input)
            loss = loss_fn(result, tgt.long())
            total_loss += loss.item()
            _, predicted = torch.max(result, 1)
            total_correct += (predicted == tgt).sum().item()
    accuracy = total_correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

def train(model, train_dataloader, test_dataloader, num_epoch, device):
    train_loss_coll = []
    train_accuracy_coll = []
    val_loss_coll = []
    val_accuracy_coll = []
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epoch):
        model = model.to(torch.float32).to(device)
        model.train()
        train_loss = 0
        total_correct = 0  # Reset total_correct at the beginning of each epoch

        for i, batch in enumerate(train_dataloader):
            input, tgt = batch[0].to(device), batch[1].to(device)
            input = input.to(torch.float32)
            tgt = tgt.to(torch.float32)
            optimizer.zero_grad()
            result = model(input)
            loss = loss_fn(result, tgt.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Add loss item instead of tensor
            _, predicted = torch.max(result, 1)
            total_correct += (predicted == tgt).sum().item()
        train_accuracy = total_correct / len(train_dataloader.dataset)  # Correct the dataset used

        adjust_learning_rate(epoch, optimizer)
        train_loss = train_loss / len(train_dataloader)
        test_loss, test_accuracy = test(model, test_dataloader, device)

        train_loss_coll.append(train_loss)
        train_accuracy_coll.append(train_accuracy)
        val_loss_coll.append(test_loss)
        val_accuracy_coll.append(test_accuracy)

        print("Epoch {}, Train loss {:.4f}, Train acc {:.4f}, Test loss {:.4f}, Test acc {:.4f}".format(
            epoch, train_loss, train_accuracy, test_loss, test_accuracy))

    return train_loss_coll, train_accuracy_coll, val_loss_coll, val_accuracy_coll

learning_rate = 0.01
num_epochs = 100
batch_size = 128


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda")

model = L1_att_block().to(device)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
start = time.time()

train_loss, train_acc, val_loss, val_acc = train(model, train_loader, val_loader, num_epochs, device)
end = time.time()

epochs_range = range(len(train_loss))
plt.plot(epochs_range, train_loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.plot(epochs_range, train_acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Val Accuracy")
plt.legend(loc='upper right')
plt.title('Train and Val Loss and Accuracy')
plt.show()
print('total time:',end-start)