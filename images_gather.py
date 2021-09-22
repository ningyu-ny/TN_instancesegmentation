import os
import cv2 as cv
def CrossOver(directory_name: str, fl):
    """
    遍历整个文件夹内的所有nii.gz文件，将子文件夹也遍历
    :param directory_name: 待遍历的文件夹名称
    :param fl: 列表
    :return: 含有nii.gz的文件列表
    """
    for i in os.listdir(directory_name):  # 遍历整个文件夹
        path = os.path.join(directory_name, i)
        if os.path.isfile(path) and (
                i.split('.')[-1]) == 'jpg':  # 判断是否为一个文件，排除文件夹
            fl.append(path)
        elif os.path.isdir(path):
            newdir = path
            CrossOver(newdir, fl)
    return fl

directory_name=r'F:\Github_data\TN_instancesegmentation\train\masks'
imageslist=[]
fl=CrossOver(directory_name,imageslist)
filelist_useful=[]
for filename in fl:
    image=cv.imread(filename)
    if image.max()>0:
        filelist_useful.append(filename)

#保存masks
for i,filename in enumerate(filelist_useful):
    image = cv.imread(filename,cv.IMREAD_GRAYSCALE)
    # savepath=filename.replace("train", "test_new")
    savepath=r'F:\Github_data\TN_instancesegmentation\test_new\masks'
    # dir_name='\\'.join(savepath.split('\\')[:-1])
    if not os.path.exists(savepath):
        os.mkdir(savepath)  # 只能建立一层！！！
    # '%s/%05d.jpg' % (savepath, i)
    cv.imwrite('%s/%05d.jpg' % (savepath, i), image)


#保存images
for i, filename in enumerate(filelist_useful):
    filename=filename.replace('masks', 'images')
    image = cv.imread(filename,cv.IMREAD_GRAYSCALE)
    savepath = r'F:\Github_data\TN_instancesegmentation\test_new\images'
    if not os.path.exists(savepath):
        os.mkdir(savepath)  # 只能建立一层！！！
    cv.imwrite('%s/%05d.jpg' % (savepath, i), image)
    # pass

print("over")