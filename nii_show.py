import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np


def showNii(img):
    for i in range(img.shape[0]):
        plt.imshow(img[i,:,:],cmap='gray')
        plt.show()


if __name__ == '__main__':

    itk_img = sitk.ReadImage('./data/mask.nii.gz')
    img = sitk.GetArrayFromImage(itk_img)
    print(img.shape)
    b,h,w=img.shape
    showNii(img[0,:,:].reshape(1,h,w))