"""VISUALIZATION UTILITIES
This example illustrates some of the utilities
for visualizing images, bounding boxes, and segmentation masks."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
# from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
from torchvision.transforms.functional import convert_image_dtype
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


image= read_image('qyc.jpg')
boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
colors = ["blue", "yellow"]
result = draw_bounding_boxes(image, boxes, colors=colors, width=5)
show(result)

# Visualizing bounding boxes



