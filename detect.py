import torch
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime
import cv2
import os


# 数据预处理
image_path = "data/test"
val_dataset = []
for i in range(len(val_dataset)):
    val_image = cv2.imread(os.path.join(image_path,'0' + str(i+100) + '.png'), cv2.IMREAD_GRAYSCALE)
    val_image = cv2.resize(val_image, (512, 512), interpolation=cv2.INTER_LINEAR) # onnx模型只接受（512*512）
    val_image = np.array(val_image)
    val_image = np.expand_dims(val_image, axis=2)
    h, w, _ = val_image.shape
    val_image = cv2.fastNlMeansDenoising(val_image, None, 10, 10, 7)# 加个滤波分割效果会好一点
    val_image = val_image.astype('float32')
    val_image = cv2.cvtColor(val_image, cv2.COLOR_GRAY2BGR)
    val_image = val_image / 255.
    val_image = np.transpose(val_image, (2, 0, 1))
    val_dataset.append(val_image)

imga = val_dataset[2]
imgorigin  = imga.copy()
imgorigin = np.transpose(imgorigin,(1,2,0)) #(512, 512, 3)

print(imgorigin.shape)
imga = np.expand_dims(imga, axis=0)

segmentation = np.zeros((512,512,1))
x_data = torch.tensor(imga, dtype=torch.float32)

model = onnxruntime.InferenceSession("out/UNet/unet.onnx") # bone


ort_inputs = {model.get_inputs()[0].name: x_data.numpy()} # x_data是个tensor 需要将其转为numpy
output = model.run(None, ort_inputs) # class list
output = output[0]

output = np.argmax(output,axis=1)


output = output.transpose(1,2,0)

for i in np.arange(512):
    for j in np.arange(512):
        if output[i,j,0] == 2:
            output[i,j,0] = 50
        if output[i,j,0] == 1:
            output[i,j,0] = 100

segmentation[:, :, 0] = output[:,:,0]
plt.figure(figsize=(3, 3),dpi=512)
segmentation = np.squeeze(segmentation)
plt.subplot(1,2,1),plt.imshow(imgorigin, 'gray'), plt.title('origin'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(segmentation,'gray'),plt.title('predict'),plt.xticks([]),plt.yticks([])
plt.show()
