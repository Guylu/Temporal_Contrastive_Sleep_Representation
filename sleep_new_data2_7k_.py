import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, auc
from torchvision import datasets, transforms
import wandb
import torch
import torchvision
import cv2
from efficientnet_pytorch import EfficientNet
import random
import os
import os
import sys
import time
import math
import random
from datetime import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from PIL import Image
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from LabData.DataLoaders import SubjectLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
# from torch.utils.data.cache import CacheDataset
from functools import lru_cache
import umap
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import umap.plot
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# remember to log everything to wandb
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

wandb.init(project="Sleep_con")  # , mode='disabled'


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, std_dev):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.std_dev = std_dev
        self.weights = self._initialize_weights()

    # def _initialize_weights(self):
    #     weights = torch.zeros(self.num_classes, self.num_classes)
    #     for i in range(self.num_classes):
    #         for j in range(self.num_classes):
    #             weights[i, j] = torch.exp(torch.tensor(-((i - j) ** 2) / (2 * self.std_dev ** 2)))
    #     return weights

    def _initialize_weights(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a 2D grid of indices, where each row and column represents a class
        x = torch.arange(self.num_classes, ).view(-1, 1)
        y = torch.arange(self.num_classes, ).view(1, -1)
        # Calculate the squared distance between each pair of indices
        dists = ((x - y) ** 2)
        # Apply the exponential function to the distances
        weights = torch.exp(-dists / (2 * self.std_dev ** 2))
        return weights.detach().to(device)

    def forward(self, logits, labels):
        # weights = torch.zeros(self.num_classes, self.num_classes, device=logits.device)
        # for i in range(self.num_classes):
        #     for j in range(self.num_classes):
        #         weights[i, j] = torch.exp(torch.tensor(-((i - j) ** 2) / (2 * self.std_dev ** 2)))
        weights = self.weights.to(logits.device)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = -(weights[labels, :] * log_probs).sum(dim=-1).mean()
        return loss


class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=3)

        # input with 7457 spatial dimensions
        self.fc1 = nn.Linear(2112, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Better1DCNN(nn.Module):
    def __init__(self, temp=0.07, std=5, dropout=0.2):
        super(Better1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=3)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # input with 7457 spatial dimensions
        self.fc1 = nn.Linear(2112, 128)
        self.fc2 = nn.Linear(128, 8)

        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / temp))
        self.std = nn.Parameter(torch.ones([]) * std)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Best1D_CNN_Found_After_Many_Experimentation(nn.Module):
    def __init__(self, dim=32, temp=0.2, std=5, p1=0.2, p2=0.2, p3=0.2, p4=0.5):
        super(Best1D_CNN_Found_After_Many_Experimentation, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=10, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p1)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=p2)

        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=p3)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc1 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=p4)

        # Output layer
        self.fc2 = nn.Linear(64, dim)

        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / temp))
        self.std = nn.Parameter(torch.ones([]) * std)

    def forward(self, x):
        # Pass input through first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Pass through second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Pass through third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Global average pooling
        x = self.pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Pass through fully connected layer
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        # Pass through output layer
        x = self.fc2(x)

        return x


class Fourrier_on_Best(nn.Module):
    def __init__(self, dim=8, temp=0.2, std=5, p1=0.2, p2=0.2, p3=0.2, p4=0.5):
        super(Fourrier_on_Best, self).__init__()
        self.best1 = Best1D_CNN_Found_After_Many_Experimentation(dim, temp, std, p1, p2, p3, p4)
        self.best2 = Best1D_CNN_Found_After_Many_Experimentation(dim, temp, std, p1, p2, p3, p4)
        self.temp1 = self.best1.temp
        self.temp2 = self.best2.temp
        self.temp3 = nn.Parameter(torch.ones([]) * np.log(1 / temp))
        self.std1 = self.best1.std
        self.std2 = self.best2.std

        # mlp with strong extraction of features
        self.MLP1 = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x):
        # try:
        #     x = torch.swapaxes(x.reshape(-1, 486, 10), 1, 2)
        # except:
        #     pass
        # is (batch, 6, 7457)
        # print("going to b1")
        x1 = self.best1(x)
        x2 = torch.abs(torch.fft.rfft(x, dim=2))
        # print("going to b2")
        x2 = self.best2(x2)

        x3 = self.MLP1(x1)
        x4 = self.MLP2(x2)
        return x1, x2, x3, x4
        # we will then do our weighted contrative loss on x1 representations (maybe on x2 somehow too)
        # and we will also do regular contrastive loss between x1 and x2 representations (on the batch dimension)


def con_loss(out1, out2, func, device, t, One=False):
    if One:
        out = torch.nn.functional.normalize(out1)
        return func(torch.matmul(out, out.T) * t.exp(), torch.arange(out.shape[0], device=out.device))
    out1 = torch.nn.functional.normalize(out1)
    out2 = torch.nn.functional.normalize(out2)
    simmilarity1 = torch.matmul(out1, out2.T)
    simmilarity2 = torch.matmul(out2, out1.T)
    loss1 = func(simmilarity1 * t.exp(), torch.arange(simmilarity1.shape[0], device=out1.device))
    loss2 = func(simmilarity2 * t.exp(), torch.arange(simmilarity2.shape[0], device=out1.device))
    return (loss1 + loss2) / 2


class CSV_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path='/net/mraid08/export/jafar/Microbiome/Analyses/guy/G_CLIP'
                      '/Contrastive_Medical_Information_Representation/MODEL_TO_END_ALL_MODELS/sleep_7k_3/'):
        self.path = path
        self.file_list = os.listdir(path)

    def __len__(self):
        return len(self.file_list) - 1

    # @lru_cache(maxsize=20)
    def __getitem__(self, idx, ret_all=False):
        # print(idx)
        data = torch.load(self.path + f'{idx}.pt')
        id = data['ids']
        data = data['data']
        if not ret_all:
            data = [np.swapaxes(data[i], 1, 2).reshape(-1, 13)[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]] for i in
                    range(3)]
            data = torch.from_numpy(np.stack(data))
            data[:, :, 7] = data[:, :, 7].clamp(min=0, max=100)
            # try:
            #     g = torch.swapaxes(data[0].reshape(-1, 486, 10), 1, 2)
            #     g = torch.swapaxes(data[1].reshape(-1, 486, 10), 1, 2)
            #     g = torch.swapaxes(data[2].reshape(-1, 486, 10), 1, 2)
            # except:
            #     pass
        return data, id


dataset = CSV_Dataset(path='/net/mraid08/export/jafar/Microbiome/Analyses/guy/G_CLIP'
                           '/Contrastive_Medical_Information_Representation/MODEL_TO_END_ALL_MODELS/sleep_7k_3/')
train_proportion = 0.9
print(f'train proportion is {train_proportion}')
wandb.log({'train_proportion': train_proportion})

train_size = int(train_proportion * len(dataset))
test_size = len(dataset) - train_size
train_dataset = torch.utils.data.Subset(dataset, [i for i in range(train_size)])
test_dataset = torch.utils.data.Subset(dataset, [i for i in range(train_size, len(dataset))])

print(f'whole dataset size: {len(dataset)}')
print(f'train size: {len(train_dataset)}')
print(f'test size: {len(test_dataset)}')

wandb.log({'dataset size': len(dataset)})
wandb.log({'train size': len(train_dataset)})
wandb.log({'test size': len(test_dataset)})

# train_dataset = CacheDataset(dataset, cache_size=20)
batch_size = 1
num_workers = 10
wandb.log({'batch_size': batch_size})
wandb.log({'num_workers': num_workers})

print(f'batch_size: {batch_size}')
print(f'num_workers: {num_workers}')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)

epochs = 50  # 125 on better
temp = 0.2  # 0.2 got on epoch 70 val loss of 200.01
std = 5
lr = 0.001
weight_decay = 0.01
p1 = 0.2
p2 = 0.2
p3 = 0.2
p4 = 0.5
dim = 32
step_size = 1
gamma = 0.99
num_classes = 7457  # train_dataset[0][0][0].shape[0]

wandb.log({'epochs': epochs})
wandb.log({'temp': temp})
wandb.log({'std': std})
wandb.log({'lr': lr})
wandb.log({'weight_decay': weight_decay})
wandb.log({'p1': p1})
wandb.log({'p2': p2})
wandb.log({'p3': p3})
wandb.log({'p4': p4})
wandb.log({'dim': dim})
wandb.log({'step_size': step_size})
wandb.log({'gamma': gamma})
wandb.log({'num_classes': num_classes})

print(f'temp: {temp}')
# model = Best1D_CNN_Found_After_Many_Experimentation(
#     temp=temp, std=5, p1=0.2, p2=0.2, p3=0.3, p4=0.5)  # Best1D_CNN_Found_After_Many_Experimentation()
model = Fourrier_on_Best(dim=dim, temp=temp, std=std, p1=p1, p2=p2, p3=p3, p4=p4)
model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
print("preparing loss function")
loss_fn = WeightedCrossEntropyLoss(num_classes=num_classes, std_dev=model.module.std1)
print("prepared")
reg_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create a folder for this run
path_name = f'/net/mraid08/export/jafar/Microbiome/Analyses/guy/G_CLIP/Contrastive_Medical_Information_Representation/MODEL_TO_END_ALL_MODELS/sleep_models/{wandb.run.name}'
os.mkdir(path_name)
if False:
    print("reloading")
    path_for_checkpoint = '/charmed-pyramid-26/20230120-200736_50_1_0.001_1_0.99_0.2_5__7_t_loss_957.0588077190512_v_loss_975.80517578125.pt'
    file = torch.load(path_for_checkpoint)
    model = Fourrier_on_Best(dim=dim, temp=temp, std=std, p1=p1, p2=p2, p3=p3, p4=p4)
    model.load_state_dict(file['model_state_dict'])
    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    loss_fn = WeightedCrossEntropyLoss(num_classes=num_classes, std_dev=model.module.std1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    optimizer.load_state_dict(file['optimizer_state_dict'])
    scheduler.load_state_dict(file['scheduler_state_dict'])

model = model.to(device)
for epoch in range(epochs):
    # print(f'\t\t\t\tEpoch {epoch}')
    train_loss_acc_time = 0
    train_loss_acc_freq = 0
    train_loss_acc_cross = 0
    per_person = []
    start = time.time()
    for i, (nights_batch, name) in enumerate(train_loader):
        if i % 20 == 0:
            print(f"\t\t\t{i + 1}/{len(train_loader)}")
        person = []
        device = 'cuda:' + str(i % torch.cuda.device_count())
        train_loss_time = 0
        train_loss_freq = 0
        train_loss_cross = 0
        for nights in nights_batch:
            nights = nights.squeeze()
            for j, night in enumerate(nights):
                # print(j)
                data = night.to(device).float()
                data = torch.swapaxes(data.reshape(-1, 486, 10), 1, 2)
                # print("going into model")
                out1, out2, out3, out4 = model(data)
                train_loss_time += con_loss(out1, out1, loss_fn, device, model.module.temp1, One=True)
                train_loss_freq += con_loss(out2, out2, loss_fn, device, model.module.temp2, One=True)
                train_loss_cross += con_loss(out3, out4, loss_fn, device, model.module.temp3)
                torch.cuda.empty_cache()
        train_loss = train_loss_time + train_loss_freq + train_loss_cross
        train_loss.backward()
        if (i + 1) % 5 == 0:
            print(
                f'\t\ttrain loss total: {train_loss.item():8.3f} train loss time: {train_loss_time.item():8.3f} '
                f'train loss freq: {train_loss_freq.item():8.3f} train loss cross: {train_loss_cross.item():8.3f}')

        #         out = torch.cat((out1, out2), dim=1)
        #         person.append(out)
        # person = torch.cat(person)
        # per_person.append(person)
        train_loss_acc_time += train_loss_time.item()
        train_loss_acc_freq += train_loss_freq.item()
        train_loss_acc_cross += train_loss_cross.item()
    # per_person = torch.stack(per_person)
    # per_person_simmilarity = torch.matmul(per_person, per_person.T)
    # # contrastove loss across all people
    # person_loss = WeightedCrossEntropyLoss(num_classes=per_pesron.shape[0], std_dev=0)
    # person_con_loss = person_loss(per_person_simmilarity * t.exp(), torch.arange(per_person_simmilarity.shape[0]))
    #
    # train_loss += person_con_loss
    train_loss_acc_time = train_loss_acc_time / len(train_dataset)
    train_loss_acc_freq = train_loss_acc_freq / len(train_dataset)
    train_loss_acc_cross = train_loss_acc_cross / len(train_dataset)

    train_loss_acc = train_loss_acc_time + train_loss_acc_freq + train_loss_acc_cross
    # print(f'Training Loss : {train_loss.item()}')
    val_loss_time = 0
    val_loss_freq = 0
    val_loss_cross = 0
    start2 = time.time()
    with torch.no_grad():
        model.eval()
        for i, (nights_batch, name) in enumerate(test_loader):
            if i % 20 == 0:
                print(f"\t\t\t{i + 1}/{len(test_loader)}")
            for nights in nights_batch:
                nights = nights.squeeze()
                for night in nights:
                    data = night.to(device).float()
                    data = torch.swapaxes(data.reshape(-1, 486, 10), 1, 2)
                    out1, out2, out3, out4 = model(data)
                    val_loss_time += con_loss(out1, out1, loss_fn, device, model.module.temp1, One=True)
                    val_loss_freq += con_loss(out2, out2, loss_fn, device, model.module.temp2, One=True)
                    val_loss_cross += con_loss(out3, out4, loss_fn, device, model.module.temp2)
        model.train()
    finish = time.time()
    # print time in minutes and seconds
    print(
        f"Time for eval  {epoch + 1} is {int((finish - start2) // 60)} minutes and {int((finish - start2) % 60)} seconds")
    val_loss_acc_time = val_loss_time.item() / len(test_dataset)
    val_loss_acc_freq = val_loss_freq.item() / len(test_dataset)
    val_loss_acc_cross = val_loss_cross.item() / len(test_dataset)
    val_loss_acc = val_loss_time + val_loss_freq + val_loss_cross
    print(
        f'Epoch {epoch + 1:3d}/{epochs} |Train Loss {train_loss_acc:8.3f}'
        f': {train_loss_acc_time:8.3f}, {train_loss_acc_freq:8.3f}, {train_loss_acc_cross:8.3f} |'
        f' Val Loss {val_loss_acc:8.3f}:'
        f' {val_loss_acc_time:8.3f}, {val_loss_acc_freq:8.3f}, {val_loss_acc_cross:8.3f} |'
        f' Temp1 {model.module.temp1.item():2.3f} | Temp2 {model.module.temp2.item():2.3f} | Temp3 {model.module.temp3.item():2.3f} |'
        f' lr {optimizer.param_groups[0]["lr"]:2.3f} | std1 {model.module.std1:2.3f} | std2 {model.module.std2:2.3f} |'
        f' time: {int((finish - start) // 60)}:{int((finish - start) % 60):02}')

    wandb.log({'train_loss': train_loss_acc})
    wandb.log({'val_loss': val_loss_acc})

    wandb.log({'train_loss_acc_time': train_loss_acc_time})
    wandb.log({'train_loss_acc_freq': train_loss_acc_freq})
    wandb.log({'train_loss_acc_cross': train_loss_acc_cross})

    wandb.log({'val_loss_time': val_loss_time})
    wandb.log({'val_loss_freq': val_loss_freq})
    wandb.log({'val_loss_cross': val_loss_cross})

    wandb.log({'temp1': model.module.temp1.item()})
    wandb.log({'temp2': model.module.temp2.item()})
    wandb.log({'temp3': model.module.temp3.item()})
    wandb.log({'lr': optimizer.param_groups[0]["lr"]})
    wandb.log({'std1': model.module.std1.item()})
    wandb.log({'std2': model.module.std2.item()})

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    if epoch % 1 == 0:
        name = f'{time.strftime("%Y%m%d-%H%M%S")}_{epochs}_{batch_size}_{lr}_{step_size}_{gamma}_{temp}_{std}__{epoch}_t_loss_{train_loss_acc}_v_loss_{val_loss_acc}'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f'{path_name}/{name}.pt')

    # loss_fn = WeightedCrossEntropyLoss(num_classes=486, std_dev=model.module.std1)
    # print()
    # print()

# create a folder named saved models (if isnt already created
# and then save the model with its name being the time it was saved and its train loss and val loss

name = f'{time.strftime("%Y%m%d-%H%M%S")}_{epochs}_{batch_size}_{lr}_{step_size}_{gamma}_{temp}_{std}__{epoch}_t_loss_{train_loss_acc}_v_loss_{val_loss_acc}'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, f'00_FIN_{path_name}_{name}.pt')
print('model saved')

raise ValueError("stop")

#

#

#


with torch.no_grad():
    data_embds = []
    sleep_stages = []
    people = []
    time = []
    for i in range(len(train_dataset)):
        print(i)
        data = train_dataset[i][0][0].to(device).float()
        c = train_dataset.dataset.__getitem__(i, ret_all=True)[0][0].mean(axis=-1)[:, 11]
        idx = len(c) - np.argmax(c[::-1] != 14) - 1
        c = c[:idx]
        c = np.round(c).astype(int)
        t = np.arange(len(c))

        out1, out2, out3, out4 = model(data)
        out1 = torch.nn.functional.normalize(out1)[:idx]
        out2 = torch.nn.functional.normalize(out2)[:idx]
        out3 = torch.nn.functional.normalize(out3)[:idx]
        out4 = torch.nn.functional.normalize(out4)[:idx]

        data_embds.append(out1)
        sleep_stages.append(c)

        p = np.ones(c.shape) * i
        people.append(p)
        time.append(t)

    sleep_stages2 = np.hstack(sleep_stages)
    data_embds2 = torch.vstack(data_embds)
    people2 = np.hstack(people)
    time2 = np.hstack(time)


def determine_k(data, max_k):
    bic = []
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k)
        gmm.fit(data)
        bic.append(gmm.bic(data))
    return bic


data = data_embds[0].cpu().numpy()
ss = sleep_stages[0]
bic = determine_k(data, 10)
plt.plot(bic)
plt.show()
# choose min index in bic
k = np.argmin(np.array(bic)) + 1
print(k)
# do kmeans with k
kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
# get labels from kmeans
labels = kmeans.labels_

embds = umap.UMAP(n_components=3, metric='cosine', n_neighbors=80, min_dist=0.4,
                  densmap=False, ).fit_transform(data_embds2.detach().cpu().numpy())
coloring = sleep_stages2  # people2 # time2 labels # sleep_stages2
fig = go.Figure(data=[go.Scatter3d(
    x=embds[:, 0],
    y=embds[:, 1],
    z=embds[:, 2],
    mode='markers',
    marker=dict(
        size=4,
        color=coloring,  # set color to an array/list of desired values
        colorscale='Viridis',  # choose a colorscale
        opacity=1
    )
)])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# carete a folder called plots (if isnt already created)
os.makedirs('plots', exist_ok=True)
# save it with time y m d h m
fig.write_html(f'plots/{time.strftime("%Y%m%d-%H%M%S")}.html')

#

#
# load model 00_FIN_20230118-162657_50_2_0.001_1_0.99_0.2_5__50_t_loss_596_v_loss_581.pt
# file = torch.load('00_FIN_20230118-162657_50_2_0.001_1_0.99_0.2_5__50_t_loss_596_v_loss_581.pt')
# model.load_state_dict(file['model_state_dict'])
data = train_dataset[0][0][0].to(device).float()
c = train_dataset.dataset.__getitem__(0, ret_all=True)[0][0].mean(axis=-1)[:, 11]
idx = len(c) - np.argmax(c[::-1] != 14) - 1
out1, out2, out3, out4 = model(data)
out1 = torch.nn.functional.normalize(out1)[:idx]
out2 = torch.nn.functional.normalize(out2)[:idx]
out3 = torch.nn.functional.normalize(out3)[:idx]
out4 = torch.nn.functional.normalize(out4)[:idx]
simmilarity1 = torch.matmul(out1, out1.T)
simmilarity2 = torch.matmul(out2, out2.T)
simmilarity3 = torch.matmul(out3, out4.T)
plt.imshow(out1.cpu().detach().numpy().T, aspect=20)
plt.title("Train - Time")
plt.show()
plt.imshow(out2.cpu().detach().numpy().T, aspect=20)
plt.title("Train - Frequency")
plt.show()
plt.imshow(np.exp(simmilarity1.cpu().detach().numpy()))
plt.title("Train - Time")
plt.show()
plt.imshow(np.exp(simmilarity2.cpu().detach().numpy()))
plt.title("Train - Frequency")
plt.show()
plt.imshow(np.exp(simmilarity3.cpu().detach().numpy()))
plt.title("Train - Time-Frequency")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(out1.detach().cpu().numpy())
plt.scatter(pca[:, 0], pca[:, 1], c=data.mean(dim=-1).cpu().numpy()[:, 11])
plt.show()

import umap

embds = umap.UMAP(n_components=2, metric='cosine', n_neighbors=80, min_dist=0.4,
                  densmap=False, ).fit_transform(out1.detach().cpu().numpy()[:idx])
plt.scatter(embds[:idx, 0], embds[:idx, 1], c=c[:idx], s=8, alpha=0.2, cmap='Set1')
plt.show()

plt.scatter(embds[:idx, 0], embds[:idx, 1], c=np.arange(idx), s=8, alpha=0.2, cmap='gist_rainbow')
plt.colorbar()
plt.show()

import umap.plot

mapper = umap.UMAP(n_components=2, metric='cosine', n_neighbors=80, min_dist=0.4,
                   densmap=False, ).fit(out1.detach().cpu().numpy()[:idx])
umap.plot.points(mapper, labels=c[:idx], theme='fire', show_legend=False)

umap.plot.points(mapper, labels=np.arange(idx), theme='fire', show_legend=False)

embds = umap.UMAP(n_components=3, metric='cosine', n_neighbors=80, min_dist=0.4,
                  densmap=False, ).fit_transform(out1.detach().cpu().numpy()[:idx])
fig = go.Figure(data=[go.Scatter3d(
    x=embds[:idx, 0],
    y=embds[:idx, 1],
    z=embds[:idx, 2],
    mode='markers',
    marker=dict(
        size=4,
        color=c[:idx],  # set color to an array/list of desired values
        colorscale='Viridis',  # choose a colorscale
        opacity=1
    )
)])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.write_html("t1.html")

d = pd.DataFrame(data.mean(axis=2).cpu().detach().numpy())
d.plot(subplots=True, legend=False)
plt.title("Train")
plt.show()

# first_sample_events = pd.read_csv('3785226862_events.csv')
# first_sample_events[::100].plot(subplots=True)
# plt.title("Train")
# plt.show()

###
# Test
data = test_dataset[0][0][0].to(device).float()
out1, out2, ou3, out4 = model(data)
out1 = torch.nn.functional.normalize(out1)
out2 = torch.nn.functional.normalize(out2)
simmilarity1 = torch.matmul(out1, out1.T)
simmilarity2 = torch.matmul(out2, out2.T)
simmilarity3 = torch.matmul(out1, out2.T)
plt.imshow(out1.cpu().detach().numpy().T, aspect=5)
plt.title("Test - Time")
plt.show()
plt.imshow(out2.cpu().detach().numpy().T, aspect=5)
plt.title("Test - Frequency")
plt.show()
plt.imshow(np.exp(simmilarity1.cpu().detach().numpy()))
plt.title("Test - Time")
plt.show()
plt.imshow(np.exp(simmilarity2.cpu().detach().numpy()))
plt.title("Test - Frequency")
plt.show()
plt.imshow(np.exp(simmilarity3.cpu().detach().numpy()))
plt.title("Test - Time-Frequency")
plt.show()

# d = pd.DataFrame(data.mean(axis=2).cpu().detach().numpy())
# d.plot(subplots=True)
# plt.title('Test')
# plt.show()

# first_sample_events = pd.read_csv('9859192071_events.csv')
# first_sample_events[::100].plot(subplots=True)
# plt.title('Test')
# plt.show()
