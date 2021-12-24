from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import albumentations as A

# def get_attention_map(img, model, transform, get_mask=False):
#     x = transform(img)
#     print(x.size())

#     logits, att_mat = model(x.unsqueeze(0))
#     print(len(att_mat))

#     att_mat = torch.stack(att_mat).squeeze(1)
#     print(att_mat.shape)

#     # Average the attention weights across all heads.
#     att_mat = torch.mean(att_mat, dim=1)
#     print(att_mat.shape)

#     # To account for residual connections, we add an identity matrix to the
#     # attention matrix and re-normalize the weights.
#     residual_att = torch.eye(att_mat.size(1))
#     aug_att_mat = att_mat + residual_att
#     aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

#     # Recursively multiply the weight matrices
#     joint_attentions = torch.zeros(aug_att_mat.size())
#     joint_attentions[0] = aug_att_mat[0]

#     for n in range(1, aug_att_mat.size(0)):
#         joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

#     v = joint_attentions[11]
#     print("*"*20)
#     print(v[0, 1:].shape)
#     grid_size = int(np.sqrt(aug_att_mat.size(-1)))
#     mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
#     print(mask.shape)
#     print(mask.max())

#     if get_mask:
#         result = cv2.resize(mask / mask.max(), img.size)
#     else:        
#         mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
#         result = (mask * img).astype("uint8")
    
#     return result

# def plot_attention_map(original_img, att_map):
#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
#     ax1.set_title('Original')
#     ax2.set_title('Attention Map 12 Layer')
#     _ = ax1.imshow(original_img)
#     _ = ax2.imshow(att_map)

#     plt.show()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=False):
        self.image_root = './images'
        self.bbox_groundtruth = './annotations/xmls'
        annotation = root
        if train:
            annotation = annotation + '/trainval.txt'
        else:
            annotation = annotation + '/test.txt'
        
        self.image_ids = []
        self.label = []
        self.transform = transform
        with open(annotation, 'r') as f:
            for line in f.readlines():
                info = line.split(' ')
                # print(info)
                if not os.path.isfile(self.bbox_groundtruth + '/' + info[0] + '.xml') and train:
                    continue
                self.image_ids.append(info[0])
                self.label.append(int(info[1])-1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_root+"/"+self.image_ids[idx]+'.jpg').convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, self.label[idx]

def loader(train_batch_size):
    image_root = './images'
    annotation_root = './annotations'

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=0.5),
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = Dataset(annotation_root, train_tfm, train=True)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_set = Dataset(annotation_root, test_tfm, train=False)
    test_loader = DataLoader(test_set, batch_size=train_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = CONFIGS["ViT-B_16"]
num_classes = 37
model = VisionTransformer(config, 224, zero_head=True, num_classes=num_classes, vis=True)
checkpoint = torch.load('./output/layer_12_checkpoint.bin')
model.load_state_dict(checkpoint)

# for name, param in model.state_dict().items():
    # print(name, param)

train_loader, test_loader = loader(train_batch_size=32)
eval_losses = AverageMeter()

all_preds, all_label = [], []
epoch_iterator = tqdm(test_loader,
                      desc="Validating... (loss=X.X)",
                      bar_format="{l_bar}{r_bar}",
                      dynamic_ncols=True,
                      disable=-1 not in [-1, 0])

loss_fct = torch.nn.CrossEntropyLoss()
model1 = model.to(device)
model1.eval()
for step, batch in enumerate(epoch_iterator):
    batch = tuple(t.to(device) for t in batch)
    x, y = batch
    with torch.no_grad():
        logits = model1(x)[0]

        eval_loss = loss_fct(logits, y)
        eval_losses.update(eval_loss.item())

        preds = torch.argmax(logits, dim=-1)

    if len(all_preds) == 0:
        all_preds.append(preds.detach().cpu().numpy())
        all_label.append(y.detach().cpu().numpy())
    else:
        all_preds[0] = np.append(
            all_preds[0], preds.detach().cpu().numpy(), axis=0
        )
        all_label[0] = np.append(
            all_label[0], y.detach().cpu().numpy(), axis=0
        )
    epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

all_preds, all_label = all_preds[0], all_label[0]
accuracy = simple_accuracy(all_preds, all_label)

print("accuracy: " + str(accuracy))


# config = CONFIGS["ViT-B_16"]
# num_classes = 37
# model = VisionTransformer(config, 224, zero_head=True, num_classes=num_classes, vis=True)
# checkpoint = torch.load('./output/original_checkpoint.bin')
# model.load_state_dict(checkpoint)
# test_tfm = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
# img1 = Image.open("./images/Abyssinian_54.jpg")
# result1 = get_attention_map(img1, model, test_tfm, True)
# plot_attention_map(img1, result1)

# image = cv2.imread("./images/Abyssinian_1.jpg")
# bbox_transform = A.Compose([
#         A.Resize(224, 224),
#         A.HorizontalFlip(p=0.5),
# ], bbox_params=A.BboxParams(format='pascal_voc'))

# bbox = [[333, 72, 425, 158, 0]] 
# # transformed = bbox_transform(image=image, bboxes=bbox)
# # image = Image.fromarray(cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB))
# # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# cv2.imshow('result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# print(transformed['bboxes'])