from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class FaceExpressionLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.train_class = os.listdir(root_dir)
        self.class_list = [os.path.join(root_dir, c) for c in self.train_class]
        self.class2label = {}
        self.label2class = {}
        for i in range(len(self.train_class)):
            self.class2label[self.train_class[i]] = i
            self.label2class[i] = self.train_class[i]
        self.imgs = []
        for i, j in enumerate(self.class_list):
            imgs = os.listdir(j)
            self.temp = [(os.path.join(j, k), i) for k in imgs]
            self.imgs += self.temp
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        pil_img = Image.open(img_path)
        img = self.transform(pil_img)
        pil_img.close()
        return img, label
        

class ExplicitExpressionLoader(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.imgs = os.listdir(root_dir)
        self.img_list = [os.path.join(root_dir, c) for c in self.imgs]
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path= self.img_list[index]
        pil_img = Image.open(img_path)
        img = self.transform(pil_img)
        pil_img.close()
        return img, self.label


class TestExpressionLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.imgs = os.listdir(root_dir)
        self.img_list = [os.path.join(root_dir, c) for c in self.imgs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path= self.img_list[index]
        pil_img = Image.open(img_path)
        img = self.transform(pil_img)
        pil_img.close()
        return img

# root_dir = './人脸表情分类/train'
# train_data = FaceExpressionLoader(root_dir, transform)