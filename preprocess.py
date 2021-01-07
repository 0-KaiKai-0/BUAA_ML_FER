import os
import torch
from PIL import Image
import numpy as np

train_root = './train'
train_class = os.listdir(train_root)
class_list = [os.path.join(train_root, c) for c in train_class]

imgs_tensor = torch.tensor([])
for _, j in enumerate(class_list):
    i = 0
    print(j)
    imgs = os.listdir(j)
    for k in imgs:
        img_path = os.path.join(j, k)
        img = Image.open(img_path)
        mat = np.array(img)
        imgs_tensor = torch.cat((imgs_tensor, torch.FloatTensor(mat).view(1, 48, 48)), 0)
        img.close()
        
print('train_imgs loaded!')
print(imgs_tensor.size())
        
test_root = './test'
imgs = os.listdir(test_root)
print(test_root)

for i in imgs:
    img_path = os.path.join(test_root, i)
    img = Image.open(img_path)
    mat = np.array(img)
    imgs_tensor = torch.cat((imgs_tensor, torch.tensor(mat).view(1, 48, 48)), 0)
    img.close()

imgs_tensor = imgs_tensor.cuda()
print(imgs_tensor.size())
print('mean: ',torch.mean(imgs_tensor))
print('std: ', torch.std(imgs_tensor))


