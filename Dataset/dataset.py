from paddle.io import Dataset
import numpy as np
import random
from numba import jit
from paddle.vision.transforms import hflip, vflip, ColorJitter
from paddle.vision.transforms.functional import rotate
import cv2


@jit(nopython=True)
def _calc(img_temp, minval, maxval):
    rows, cols = img_temp.shape
    for i in np.arange(rows):
        for j in np.arange(cols):
            # 避免除以0的报错
            if maxval - minval == 0:
                result = 1
            else:
                result = maxval - minval
            img_temp[i, j] = int((img_temp[i, j] - minval) / result * 255)
    return img_temp

@jit(nopython=True)
def _calclabel(img_label):
    rows, cols = img_label.shape
    img_temp = np.zeros(img_label.shape, np.uint8)
    img_mask = np.zeros(img_label.shape, np.uint8)
    for i in np.arange(rows):
        for j in np.arange(cols):
            if img_label[i, j] == 255:  # 牙齿255
                img_mask[i, j] = 1
                img_temp[i, j] = 1
            if img_label[i, j] == 127:  # 骨头 127
                img_mask[i, j] = 2
                img_temp[i, j] = 2
    return (img_temp, img_mask)

class MyDataset(Dataset):
    def __init__(self, mode='train', txt_file=None, transforms=None):
        super(MyDataset, self).__init__()
        self.mode = mode.lower()
        self.txt_file = txt_file
        self.lines = []
        self.seed = 2020
        self.transforms = transforms
        if self.mode == 'train':
            if self.txt_file is None:
                raise ValueError('train_txt cannot be empty ')
            self.lines = self.get_img_info(self.txt_file)
        elif self.mode == 'val':
            if self.txt_file is None:
                raise ValueError('val_txt cannot be empty ')
            self.lines = self.get_img_info(self.txt_file)
        else:
            raise ValueError('mode must be "train" or "val"')

    def get_img_info(self, txt_file):
        # 读取txt文档
        lines = list()
        with open(txt_file, 'r') as f:
            for line in f:
                imgA = line.split()[0].split()[0]
                imgB = line.split()[1].split()[0]
                lines.append((imgA, imgB))
        random.Random(self.seed).shuffle(lines)
        return lines

    def __getitem__(self, idx):
        # 加载原始图像
        imgA = cv2.imread(self.lines[idx][0], cv2.IMREAD_GRAYSCALE)
        imgA = cv2.resize(imgA,(512,512), interpolation=cv2.INTER_LINEAR)
        imgB, imgBmask = self._change_label_value(self.lines[idx][1])
        imgA, imgB = self.data_transform(imgA, imgB, imgBmask=None)
        return imgA, imgB

    def data_transform(self, imgA, imgB, imgBmask=None):
        imgA = np.array(imgA)
        imgB = np.array(imgB)

        imgA = np.expand_dims(imgA, axis=2)
        imgB = np.expand_dims(imgB, axis=2)

        h, w, _ = imgA.shape
        imgA = cv2.fastNlMeansDenoising(imgA, None, 10, 10, 7)
        imgA = imgA.astype('float32')
        imgA = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
        if self.mode == 'train':
            if imgBmask is not None:
                imgBmask = np.array(imgBmask)
                imgBmask = np.expand_dims(imgBmask, axis=2)
                imgA = imgA * imgBmask
            if random.random() > 0.5:
                # 随机旋转
                angle = random.randint(0, 60)
                imgA = rotate(imgA, angle)
                imgB = rotate(imgB, angle)
            # 随机水平翻转 和垂直翻转
            if random.random() > 0.5:
                imgA = hflip(imgA)
                imgB = hflip(imgB)
            if random.random() > 0.5:
                imgA = vflip(imgA)
                imgB = vflip(imgB)
            if len(imgB.shape) == 2:
                # imgA = np.expand_dims(imgA, axis=2)
                imgB = np.expand_dims(imgB, axis=2)
            if random.random() > 0.5:
                # 随机调整图像的亮度，对比度，饱和度和色调
                val = round(random.random() / 3, 1)
                color = ColorJitter(val, val, val, val)
                imgA = imgA.astype('uint8')
                imgA = color(imgA)
                imgA = imgA.astype('float32')
            if random.random() > 0.2:
                # 随机生成4个小黑色方块遮挡
                for i in range(4):
                    black_width = 50
                    black_height = 50
                    width, height, _ = imgA.shape
                    loc1 = random.randint(0, (width - black_width - 1))
                    loc2 = random.randint(0, (height - black_width - 1))
                    imgA[loc1:loc1 + black_width, loc2:loc2 + black_height:, :] = 0
                    imgB[loc1:loc1 + black_width, loc2:loc2 + black_height:, :] = 0

        imgA = imgA / 255.
        # from [H,W] to [C,H,W]
        imgA = np.transpose(imgA, (2, 0, 1))
        imgB = np.transpose(imgB, (2, 0, 1))

        imgB = imgB.astype('int64')
        return (imgA, imgB)


    def _change_label_value(self, label_path):
        # load png
        img_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        img_label = cv2.resize(img_label,(512,512), interpolation=cv2.INTER_LINEAR)
        # 根据label的dicom数据中，CT值不同，区分标签，并生成mask标签
        img_temp, img_mask = _calclabel(img_label)
        return (img_temp, img_mask)

    def __len__(self):
        return len(self.lines)

