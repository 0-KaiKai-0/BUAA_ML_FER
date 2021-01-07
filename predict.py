from torch.utils.data import DataLoader, Dataset, random_split
import os
import csv
import torch
import matplotlib as plt
import numpy as np
from torchvision import transforms
from Data import *
from resnet import *
from inception import *


transform_test = transforms.Compose([transforms.Grayscale(1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.15,),(0.4,))])
# transform = transforms.ToTensor()
# 0.685: batch_size=56
batch_size = 63

test_root = './test'
test_dataset = TestExpressionLoader(root_dir=test_root, transform=transform_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
imgs_list = os.listdir(test_root)
imgs_dict = {}
for i in range(len(imgs_list)):
    imgs_dict[imgs_list[i]] = i


layer_score_weight = [0.1, 0.2, 0.3, 0.4]
# model = Net(num_classes=7)
model = resnet34(num_classes=7)
print('before device')
device = 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('before to device', device)
# device = "cpu"
model.to(device)
print('after device')


def test(label2class, filename):
    pred_labels = torch.LongTensor([]).to(device)
    for data in test_loader:
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
            
        # outputs1, outputs2, outputs3, outputs4 = model(inputs)
        # outputs = outputs1 * layer_score_weight[0] + outputs2 * layer_score_weight[1] + outputs3 * layer_score_weight[2] + outputs4 * layer_score_weight[3]
        
        _, predicted = torch.max(outputs.data, dim=1)
        pred_labels = torch.cat((pred_labels, predicted), 0)
    print('after predict, before load')
    r = csv.reader(open(filename, 'r'))
    lines = list(r)
    for i in range(1, len(lines)):
        lines[i][1] = label2class[pred_labels[imgs_dict[lines[i][0]]].item()]
    w = csv.writer(open(filename, 'w'))
    for line in lines:
        w.writerow(line)


if __name__ == '__main__':
    print("before checkpoint")
    checkpoint = torch.load('checkpoint25.0')
    print("after checkpoint")
    model.load_state_dict(checkpoint['model'])
    class2label = checkpoint['class2label']
    label2class = checkpoint['label2class']
    batch_size = checkpoint['batch_size']
    filename = 'submission.csv'
    print('before test')
    
    with torch.no_grad():
        test(label2class, filename)