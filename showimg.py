import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def showmask(image_path, mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    img_np = np.asarray(image)
    mask_np = np.asarray(mask)
    img_rgb = image.convert("RGB")  # PIL.Image
    img_rgb_np = np.asarray(img_rgb)
    # image_gray = image.convert('L')
    # img_gray_np = np.asarray(image_gray)  # PIL.Image
    mask.putpalette([
        0, 0, 0,  # black background
        255, 0, 0,  # index 1 is red
        255, 255, 0,  # index 2 is yellow
        255, 153, 0,  # index 3 is orange
    ])
    plt.figure(figsize=(7, 8))  # 设置窗口大小
    plt.suptitle('Multi_Image')  # 窗口名称

    plt.subplot(2, 2, 1), plt.title('image')
    plt.imshow(img_np), plt.axis('off')

    plt.subplot(2, 2, 2), plt.title('image_gray cmap=gray')
    plt.imshow(image, cmap='gray'), plt.axis('off')  # 显示灰度图要加cmap

    plt.subplot(2, 2, 3), plt.title('mask')
    plt.imshow(mask), plt.axis('off')

    plt.subplot(2, 2, 4), plt.title('without cmap=gray')  # 显示灰度图未加cmap
    plt.imshow(image), plt.axis('off')


def drawRectangle(image_path,mask_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    img_np = np.asarray(image)
    mask_np = np.asarray(mask)
    img_rgb = image.convert("RGB")  # PIL.Image
    img_rgb_np = np.asarray(img_rgb)
    image_gray = image.convert('L')
    img_gray_np = np.asarray(image_gray)  # PIL.Image
    mask.putpalette([
        0, 0, 0,  # black background
        255, 0, 0,  # index 1 is red
        255, 255, 0,  # index 2 is yellow
        255, 153, 0,  # index 3 is orange
    ])
    #cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (B,G,R), Thickness)
    # img=cv2.imread(mask_path)
    # img = Image.open(image_path).convert("RGB")
    # note that we haven't converted the mask to RGB,
    # because each color corresponds to a different instance with 0 being
    # background
    # mask_show.show()
    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of binary masks
    masks = mask == obj_ids[:, None, None]

    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    obj_color=[
        0, 0, 0,  # black background
        255, 0, 0,  # index 1 is red
        255, 255, 0,  # index 2 is yellow
        255, 153, 0,  # index 3 is orange
    ]

    img = cv2.imread(image_path)
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(obj_color[3*(i+1):3*(i+2)]), 3)

    cv2.imshow('image',img)
    # cv2.imwrite('new_img.jpg', img)


if __name__ == '__main__':
    img = Image.open(os.path.join('train_data', 'images.jpg'))
    mask = Image.open(os.path.join('./train_data', 'masks', '0000.jpg'))

    image_path=os.path.join('train_data', 'images.jpg')
    image_path='train_data/images/0000.jpg'
    mask_path=os.path.join('train_data', 'masks', '0000.jpg')

    showmask(image_path, mask_path)
    drawRectangle(image_path, mask_path)
