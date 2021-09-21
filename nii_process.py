"""Nii转png，Nii转dcm，Dcm转png"""
from matplotlib import pylab as plt
from nibabel.viewers import OrthoSlicer3D
from matplotlib import pyplot as plt
import imageio
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os
import cv2
import scipy.misc


def hu_to_grayscale(volume, hu_min=None, hu_max=None, mn=None, mx=None):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    if mx is not None:
        mxval = mx
    else:
        mxval = np.max(volume)
    if mn is not None:
        mnval = mn
    else:
        mnval = np.min(volume)
    # mxval = mnval+hu_max-hu_min
    # mnval = hu_min
    # print(mxval)
    # print(mnval)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255 * im_volume
    # 转uint8+增加维度，因为标记如果是彩色则需要RGB三通道，与之相加的CT图也要存成三维数组
    return im_volume


def array2png(
    ArrayDicom, dirname, nickname="", trans=(
        0, 1, 2), size=(
            512, 512), start=None, end=None):
    dir = "./" + dirname
    if (os.path.exists(dir) == False):
        os.makedirs(dir)
    ArrayDicom = hu_to_grayscale(ArrayDicom)

    ArrayDicom = np.transpose(ArrayDicom, trans)
    if start is None:
        start = 0
    if end is None:
        end = len(ArrayDicom[0, 0])
    # ArrayDicom = hu_to_grayscale(ArrayDicom)

    for i in range(end - start):

        if ArrayDicom[:, :, start + i].shape != size:
            # cv2.resize(ArrayDicom[:, :,start + i],(128,128),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('%s/%05d%s.jpg' %
                        (dir, i, nickname), cv2.resize(ArrayDicom[:, :, start +
                                                                  i], size, interpolation=cv2.INTER_CUBIC))
            # print("Output:"+'%s/%05d.png'%(dir,i))
            # cv2.imwrite('%s/%05d_mask.png'%(dir,i), cv2.resize(ArrayDicom_mask[:, :,start + i],(128,128),interpolation=cv2.INTER_CUBIC))
            # print("Output:"+'%s/%05d_mask.png'%(dir,i))
        else:
            cv2.imwrite('%s/%05d%s.jpg' %
                        (dir, i, nickname), ArrayDicom[:, :, start + i])
            # print("Output:"+'%s/%05d.png'%(dir,i))
            # cv2.imwrite('%s/%05d_mask.png'%(dir,i), ArrayDicom_mask[:, :,start + i])
            # print("Output:"+'%s/%05d_mask.png'%(dir,i))


def nii2array(Pathnii):

    img = sitk.ReadImage(Pathnii)
    data = sitk.GetArrayFromImage(img)
    # print("模型形状",data.shape)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    return data, spacing, origin

# 遍历整个文件夹内的所有nii.gz文件，将子文件夹也遍历


def CrossOver(dir, fl):
    for i in os.listdir(dir):  # 遍历整个文件夹
        path = os.path.join(dir, i)
        if os.path.isfile(path) and (
                i.split('.')[-1]) == 'gz':  # 判断是否为一个文件，排除文件夹
            fl.append(path)
        elif os.path.isdir(path):
            newdir = path
            CrossOver(newdir, fl)
    return fl


def nii2png_single(nii_path, IsData=True):
    ori_data = sitk.ReadImage(nii_path)  # 读取一个数据
    data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array
    if IsData:  # 过滤掉其他无关的组织，标签不需要这步骤
        data1[data1 > 250] = 250
        data1[data1 < -250] = -250
    img_name = os.path.split(nii_path)  # 分离文件名
    img_name = img_name[-1]
    img_name = img_name.split('.')
    img_name = img_name[0]
    i = data1.shape[0]
    png_path = './png/single_png'  # 图片保存位置
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    for j in range(0, i):  # 将每一张切片都转为png
        if IsData:  # 数据
            # 归一化
            slice_i = (data1[j, :, :] - data1[j, :, :].min()) / \
                (data1[j, :, :].max() - data1[j, :, :].min()) * 255
            cv2.imwrite(
                "%s/%s-%d.png" %
                (png_path, img_name, j), slice_i)  # 保存
        else:   # 标签
            slice_i = data1[j, :, :] * 122
            cv2.imwrite(
                "%s/%s-%d.png" %
                (png_path, img_name, j), slice_i)  # 保存


def read_niifile(niifilepath):  # 读取niifile文件
    # img = nib.load(niifilepath)  # 下载niifile文件（其实是提取文件）
    # img_fdata = img.get_fdata()  # 获取niifile数据
    ori_data = sitk.ReadImage(nii_path)
    data = sitk.GetArrayFromImage(ori_data)
    spacing = ori_data.GetSpacing()
    origin = ori_data.GetOrigin()

    return data,spacing,origin



def save_fig(niifilepath, savepath):  # 保存为图片
    fdata = read_niifile(niifilepath)  # 调用上面的函数，获得数据
    (b, h, w) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量，第四维）
    for k in range(b):
        silce = fdata[k, :, :]  # 三个位置表示三个不同角度的切片
        imageio.imwrite(os.path.join(savepath, '{}.png'.format(k)), silce)
        # 将切片信息保存为png格式


if __name__ == '__main__':
    directory = r"F:\Github_data\Thyroid_nodule_segmentation_Unet_lvyi/data1"  # 文件夹名称
    filelist = []
    inputlist = []
    masklist = []
    output = CrossOver(directory, filelist)
    for nii_path in output:
        img_name = os.path.split(nii_path)  # 分离文件名
        img_name = img_name[-1]
        img_name = img_name.split('.')
        img_name = img_name[0]
        if img_name == 'input':
            inputlist.append(nii_path)
        elif img_name == 'mask':
            masklist.append(nii_path)

    for i, (imagename, maskname) in enumerate(zip(inputlist, masklist)):
    # for i,filename in enumerate(inputlist,start=1):
        img_nii_path = imagename
        mask_nii_path = maskname

        img_name = os.path.split(img_nii_path)
        dir_name='train'+'\\'+img_name[0].split('\\')[-1]
        data_array,_,_ = read_niifile(img_nii_path)
        data_array = data_array[:, 117:437, 126:574]  # 裁剪图片 (n,h,w) 其中h，w对应y,x
        array2png(data_array, "%s" % (dir_name), trans=(1, 2, 0), size=(448, 320))


        # img_name = os.path.split(nii_path)  # 分离文件名
        # img_name = img_name[-1]
        # img_name = img_name.split('.')
        # img_name = img_name[0]

    # nii_path=r"./data"
    # savepath=r"./data/tn"
    #     if not os.path.exists(savepath):
    #         os.makedirs(savepath)
    #     save_fig(nii_path, savepath)
    # nii2png_single(nii_path)
