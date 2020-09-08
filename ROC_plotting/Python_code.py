from resnet import resnet18

import os
from PIL import Image
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
#matplotlib inline

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from resnet import resnet18
from tensorboardX import SummaryWriter
from radam import RAdam

from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

model = resnet18(num_classes=1)
model.load_state_dict(torch.load('net_best.pt', map_location='cpu'))
model.eval()


#create mask
mask2 = np.ones((400, 400), dtype=np.uint8)
for i in range(400):
    for j in range(400):
        if ((i - 200)**2 + (j - 200)**2) < 122**2:
            mask2[i, j] = 0
plt.imshow(mask2)


class Planck(Dataset):
    def __init__(self, root='./data/train', split='None'):
        self.frequencies = ['100', '143', '217', '353', '545']
        
        # 70/15/15
        train_indices, test_indices = train_test_split(range(1000), test_size=0.3, random_state=9)
        val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=9)
        
        if split == 'train':
            indices = train_indices
        elif split == 'val':
            indices = val_indices
        elif split == 'test':
            indices = test_indices
        else:
            raise Exception()
        self.is_train = split == 'train'
        
        names_effect = list([x.split('_')[0] for x in os.listdir(f'{root}/effect/pl100/')])
        names_no_effect = list([x.split('_')[0] for x in os.listdir(f'{root}/no_effect/s100_rand/')])
        
        self.images = []
        for idx in indices:
            name_effect = names_effect[idx]
            imgs_effect = [self.open_image(f'{root}/effect/pl{f}/{name_effect}_{f}_gr.gif')
                           for f in self.frequencies]
            self.images.append((imgs_effect, 1))
            
            name_no_effect = names_no_effect[idx]
            imgs_no_effects = [self.open_image(f'{root}/no_effect/s{f}_rand/{name_no_effect}_{f}_gr.gif')
                               for f in self.frequencies]
            self.images.append((imgs_no_effects, 0))

    def __len__(self):
        return len(self.images)
    
    def open_image(self, path):
        img = Image.open(path)
        img = np.asarray(img, dtype=np.uint8).copy()
        img[mask2 == 1] = 255
        img = Image.fromarray(img)
        return img
        
    def prepare(self, imgs, angle, x_shift, y_shift):
        features = []
        for img in imgs:
            img = img.resize((96, 96), Image.NEAREST)
            if self.is_train:
                img = img.rotate(angle)
            img = (np.asarray(img, dtype=np.float32) / 255)
            if self.is_train:
                img = np.roll(img, x_shift, axis=0)
                img = np.roll(img, y_shift, axis=1)
            img = img[16:-16, 16:-16]
            mask_img = img == 1
            img[mask_img] = -1
            features.append(img)
        return np.stack(features, axis=0)
    
    def __getitem__(self, idx):
        imgs, answer = self.images[idx]
        angle = np.random.randint(0, 360)
        x_shift, y_shift = np.random.randint(-5, 6), np.random.randint(-5, 5)
        X = self.prepare(imgs, angle, x_shift, y_shift)
        return X, answer
        
def accuracy(y_pred, y_true, threshold=0.5):
    return (torch.abs(y_true - (F.sigmoid(y_pred) > threshold).float()) < 0.5).float().mean().item()


test_dataset = Planck(split='test')

y_true, y_pred = [], []
for X_sample, y_true_sample in test_dataset:
    un = torch.from_numpy(X_sample).unsqueeze(0)
    y_pred.append(model(un)[0].detach().numpy())
    y_true.append(y_true_sample)
y_true, y_pred = np.array(y_true), np.array(y_pred)
score = roc_auc_score(y_true, y_pred)
print(score)

#y_pred = y_pred.detach().numpy()
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('a.png')
