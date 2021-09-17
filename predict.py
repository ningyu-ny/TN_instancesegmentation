import utils
import torch
import glob
import os
import re
from datasetload import ThyroidNoduleDataset,get_transform
from model import get_instance_segmentation_model
# from tensorboardX import SummaryWriter
from torchvision.utils import *
import visualize
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

pth_path="./save_weights/Maskrcnn-model.pth"#模型保存路径

# dataset = ThyroidNoduleDataset('train', get_transform(train=True))
dataset_test = ThyroidNoduleDataset('test', get_transform(train=False))
torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-100])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

data_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=3, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=1, shuffle=False, num_workers=0,
#     collate_fn=utils.collate_fn)

# images,targets = next(iter(data_loader))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
model.to(device)
prefix, ext = os.path.splitext(pth_path)
ckpts = glob.glob(prefix + "-*" + ext)
ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))

if ckpts:
    checkpoint = torch.load(ckpts[-1], map_location=device)  # load last checkpoint
    model.load_state_dict(checkpoint["model"])


# for p in model.parameters():
#     p.requires_grad_(False)

model.eval()
iters = 20
for i, (image, target) in enumerate(dataset_test):
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        image.unsqueeze_(0)
        result = model(image)
        result = result[0]
        # ====================method 1========================================================================
        # result_masks = result['masks']
        # image= convert_image_dtype(image, dtype=torch.uint8).cpu()
        # score_threshold = .8
        # colors = ["red", "yellow","blue"]
        # image_with_boxes = [
        #     draw_bounding_boxes(image.squeeze(0), boxes=result['boxes'][result['scores'] > score_threshold],colors=colors, width=2)
        #
        # ]  # image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        # inst_classes = [
        #     '__background__', 'nodule'
        # ]
        # inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}
        # proba_threshold = 0.8
        # result_bool_masks = result['masks'] > proba_threshold
        # result_bool_masks = result_bool_masks.squeeze(1)
        # =========================method 2===================================================================
        plt.figure(figsize=(10, 8))
        visualize.show(image, result,position=1)
        visualize.show(image,target,position=2)

        if i >= iters - 1:
            break
