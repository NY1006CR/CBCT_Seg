from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cbct_seg.Dataset.dataset import MyDataset

train_dataset = MyDataset(mode='train', txt_file ='work/train_list.txt')
val_dataset = MyDataset(mode='val', txt_file ='work/val_list.txt')
test_dataset = MyDataset(mode='val', txt_file ='work/test_list.txt')
print('=============train dataset=============')
imga, imgb = train_dataset[4]
#imga, imgb = test_dataset[0]

imga= (imga[0])*255
imga = Image.fromarray(np.int8(imga))

imgb= (imgb[0])*255
imgb = Image.fromarray(np.int8(imgb))

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1),plt.xticks([]),plt.yticks([]),plt.imshow(imga)
plt.subplot(1,2,2),plt.xticks([]),plt.yticks([]),plt.imshow(imgb)
plt.show()