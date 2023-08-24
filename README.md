## 基于paddlepaddle的CBCT图像分割
#### 数据

------

我们用3D Slicer对原始CBCT图像进行了标注，整理后得到了700余套图像（私人数据暂不公开），然后将其整理为以下格式，所有文件都是.png格式

数据集结构：

```python
data
│
└───train
|    └───origin
|			└───01(image.jpg)
|			└───02
|    		...
|    └───label
|			└───01
|			└───02
|    		...
└───val
|    └───01
|		  └───origin
|    	  └───label
|	  ...
└───test
|    └───01
|		  └───origin
|    	  └───label
|	  ...
```

#### 训练

------

训练在一块NVIDIA TITAN Xp 12G的显卡上

```python
Python >= 3.6
```

网络是基于paddle库开发的，我们参考[paddle](https://www.paddlepaddle.org.cn/)进行安装

```python
python -m pip install paddlepaddle-gpu==2.5.1.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

便于跨平台，我们将最后的模型转存为onnx格式，需要引入onnx和onnxruntime库

```python
pip install onnxruntime
```

##### 训练策略

- 我们最初使用了预热训练，主要分为两个阶段；第一阶段先使用较小的学习率（5e-5）进行5个epoch的训练，第二阶段使用区间为[2e-4, 1.25e-5]的递减学习率进行训练，但之后发现预热策略对小样本数据集的作用不是很大。所以最后选用2e-4作为初始学习率，每10个epoch进行递减到1.25e-5
- 损失函数选用dice和交叉熵的加权结合（最初尝试了Focal loss但效果不理想）
- Batch_size为4，epoch为100

#### 实验结果

------

预测在一块NVIDIA GeForce GTX 1650 4G显卡上

我们的测试集是选用了训练集以外的20张图像（尽可能包含了多种形态的特征）进行测试，最后取平均值

| 网络模型                                                     | 平均IOU | 模型参数 | 平均推理时间 |
| ------------------------------------------------------------ | ------- | -------- | ------------ |
| U2Net                                                        | 0.8662  | 168MB    | 3.634s       |
| UNet                                                         | 0.8834  | 51.1MB   | 1.615s       |
| UNet++                                                       | 0.8826  | 31.9MB   | 1.623s       |
| [PMRID-UNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf) | 0.8365  | 9.74MB   | 1.003s       |

![image](https://github.com/NY1006CR/CBCT_Seg/assets/40394910/e4c71784-3a39-4e89-9f49-dbfa5cbe45df)

#### 可能的改进

------



- 为了加速网络收敛速度将模型的输入和输出都设置为（512,512），得到模型输出（512,512）需要恢复到原始尺寸（750,750）或（800,800），该项目用到的方法为双线性插值，这是一个可以优化改进的点，可以尝试用超分辨率重建来替代上采样
- 为了让模型有更强的泛化能力，尝试将效果最好的几个模型参数进行融合，尝试之后虽有提升但并不明显
