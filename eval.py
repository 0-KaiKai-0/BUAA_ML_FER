from torch.utils.data import DataLoader, Dataset, random_split
import os
import torch
import matplotlib as plt
import numpy as np
from torchvision import transforms
from Data import *
from resnet import *
from inception import *
from loss import *

# t1: transforms.Normalize((0.15,),(0.4,))
# t2: transforms.Normalize((0.5074,),(0.2552,))
# transform = transforms.Compose([transforms.RandomCrop(48, padding=2),
#                                 transforms.ColorJitter(brightness=0.5, contrast=0.5),
#                                 transforms.RandomHorizontalFlip(0.5),
#                                 transforms.Grayscale(1),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.15,),(0.4,))])

transform = transforms.Compose([transforms.Grayscale(1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.15,),(0.4,))])
transform_test = transforms.Compose([transforms.Grayscale(1),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.15,),(0.4,))])
# transform = transforms.ToTensor()
batch_size = 63
print('batch size: ', batch_size)

train_root = './train0'
train_dataset = FaceExpressionLoader(root_dir=train_root, transform=transform)
class2label = train_dataset.class2label
label2class = train_dataset.label2class
    
print(train_dataset.__len__())
train_dataset, eval_dataset = random_split(train_dataset, [24000, train_dataset.__len__()-24000])
eval_dataset.transform = transform_test

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)


# model = Net(num_classes=7)
layer_score_weight = [0, 0, 0, 1]
model = resnet34(num_classes=7)
print('resnet34')
print("layer_score_weight:, ", layer_score_weight)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model.to(device)

# weight = torch.FloatTensor([3995, 436, 4097, 7215, 4965, 4830, 3171]).to(device)
# weight = weight.sum() / weight
weight = torch.FloatTensor([0.4, 1, 0.4, 0.3, 0.35, 0.35, 0.5]).to(device)
lr = 0.01
weight_decay=0.0001
smoothing=0.05
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion1 = torch.nn.CrossEntropyLoss(weight=weight)
criterion2 = LabelSmoothingLoss(classes=7, smoothing=smoothing)


def train(epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs1, outputs2, outputs3, outputs4 = model(inputs)
        outputs = outputs1 * layer_score_weight[0] + outputs2 * layer_score_weight[1] + outputs3 * layer_score_weight[2] + outputs4 * layer_score_weight[3]
        _, predicted = torch.max(outputs.data, dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

        # loss = criterion1(outputs, target) + criterion2(outputs, target)
        loss = criterion2(outputs, target)
        # loss = criterion1(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch, batch_idx + 1, running_loss / 200))
            running_loss = 0.0
        
        # wrong_img = torch.tensor([]).cuda()
        # wrong_lab = torch.LongTensor([]).cuda()
        # for i in range(target.size(0)):
        #     if target[i] != predicted[i]:
        #         wrong_img = torch.cat((wrong_img, inputs[i]), 0)
        #         wrong_lab = torch.cat((wrong_lab, torch.tensor([target[i]]).cuda()), 0)
        # wrong_img = wrong_img.view(wrong_img.size(0), 1, wrong_img.size(1), wrong_img.size(2))
        # optimizer.zero_grad()
        # _, outputs = model(wrong_img)
        # loss = criterion(outputs, wrong_lab)
        # loss.backward()
        # optimizer.step()
    print ('Accuracy on train set: %d %%' % (100 * correct / total))
    

def test():
    correct = 0
    total = 0
    for data in eval_loader:
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        outputs1, outputs2, outputs3, outputs4 = model(inputs)
        outputs = outputs1 * layer_score_weight[0] + outputs2 * layer_score_weight[1] + outputs3 * layer_score_weight[2] + outputs4 * layer_score_weight[3]
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print ('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    max_epoch = 100
    print('contrast')
    # print('dropout')
    print('train0')
    # print('CrossEntropyLoss')
    # print('CrossEntropyLoss + LabelSmoothingLoss')
    print('LabelSmoothingLoss')
    print('smoothing: ', smoothing)
    print('max_epoch: ', max_epoch)
    print('LossWeight: ', weight)
    print('lr: ', lr)
    print('weight_decay: ', weight_decay)
    for epoch in range(1, max_epoch):
        train(epoch)
        if epoch % 10 == 0:
            lr = lr * 0.4
            print('epoch: ', epoch)
            print('lr:', lr)
            for param in optimizer.param_groups:
                param['lr'] = lr
        with torch.no_grad():
            test()
    state = {'model': model.state_dict(),
            'class2label': class2label,
            'label2class': label2class,
            'batch_size': batch_size}
    torch.save(state, 'checkpoint_eval3')
    