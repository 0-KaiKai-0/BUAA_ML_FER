from torch.utils.data import DataLoader, Dataset, random_split
import os
import torch
import torch.nn.functional as F
import matplotlib as plt
import numpy as np
from torchvision import transforms
from Data import *
from resnet import *
from inception import *

transform = transforms.Compose([transforms.Grayscale(1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])
# transform = transforms.ToTensor()
batch_size = 21
a = 0.5
    

model = resnet18(num_classes=7)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model.to(device)

def test(feat_centers):
    print(a)
    flag = True
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            _, outputs = model(inputs)
            outputs0 = F.normalize(outputs, p=2, dim=1)
            outputs0 = outputs0.mm(feat_centers)
            outputs = F.softmax(outputs)
            outputs = a * outputs0 + (1 - a) * outputs
            _, predicted = torch.max(outputs.data, dim=1)
            if flag:
                print(outputs0)
                print(outputs)
                print(target)
                print(predicted)
                flag = False
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print ('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    checkpoint = torch.load('checkpoint')
    model.load_state_dict(checkpoint['model'])
    eval_loader = checkpoint['eval_loader']
    class2label = checkpoint['class2label']
    label2class = checkpoint['label2class']
    batch_size = checkpoint['batch_size']
    feat_centers = checkpoint['feat_centers']
    
    with torch.no_grad():
        test(feat_centers.t())