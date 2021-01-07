from torch.utils.data import DataLoader, Dataset, random_split
import os
import torch
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

train_root = './train'
train_dataset = FaceExpressionLoader(root_dir=train_root, transform=transform)
class2label = train_dataset.class2label
label2class = train_dataset.label2class

expressionType = os.listdir(train_root)
typeDataLoaders = []
for type in expressionType:
    root_dir = os.path.join(train_root, type)
    typeDataset = ExplicitExpressionLoader(root_dir=root_dir, transform=transform, label=class2label[type])
    typeDataLoaders.append(DataLoader(typeDataset, batch_size=1))
    
train_dataset, eval_dataset = random_split(train_dataset, [24000, 4709])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)


# model = Net(num_classes=7)
model = resnet18(num_classes=7)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


def train(epoch):
    running_loss = 0.0
    print(epoch)
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        if batch_idx == 0:
            print(inputs.size())
            print(target[0])
        optimizer.zero_grad()

        _, outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 200))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs0, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print ('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
    with torch.no_grad():
        feat_center_list = []
        for dataloader in typeDataLoaders:
            feats = torch.tensor([]).to(device)
            for _, data in enumerate(dataloader):
                inputs, target = data
                inputs, target = inputs.to(device), target.to(device)
                # print(inputs.size())
                # print(target[0])
                __, outputs = model(inputs)
                feats = torch.cat((feats, outputs), 0)
            feat_center_list.append(torch.mean(feats, 0))
        feat_centers = torch.stack(feat_center_list, 0)
        print(feat_centers.size())
        feat_centers = torch.nn.functional.normalize(feat_centers, p=2, dim=1)
    state = {'model': model.state_dict(), 'eval_loader': eval_loader,
            'class2label': class2label, 'label2class': label2class,
            'batch_size': batch_size, 'feat_centers': feat_centers}
    torch.save(state, 'checkpoint')