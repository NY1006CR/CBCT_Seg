# 原始数据训练
# 师生迭代训练
import paddle
import paddle.fluid as fluid
import random
import numpy as np
import cv2


num_classes = 3
val_acc_history = []
val_loss_history = []
epoch_num = 21
batch_size = 2
count = 0
# # train
# train_dataset = MyDataset(mode='train', txt_file='work/train_list.txt')
# val_dataset = MyDataset(mode='val', txt_file='work/val_list.txt')

# fine_tone
train_dataset = MyDataset(mode='train', txt_file='work/train_list.txt')
val_dataset = MyDataset(mode='val', txt_file='work/val_list.txt')


def create_loss(predict, label, num_classes=num_classes):
    ''' 创建loss，结合dice和交叉熵 '''
    predict = paddle.transpose(predict, perm=[0, 2, 3, 1])  # shape[1, 512, 512, 2]
    predict = paddle.reshape(predict, shape=[-1, num_classes])  # [num, 2]
    m = paddle.nn.Softmax()
    predict = m(predict)
    ce_loss = paddle.nn.loss.CrossEntropyLoss(ignore_index=0, reduction='mean')
    # input 形状为 [N,C] , 其中 C 为类别数   label数据类型为int64。其形状为 [N]
    ce_loss = ce_loss(predict, label)  # 计算交叉熵
    dice_loss = fluid.layers.dice_loss(predict, label)  # 计算dice loss
    return fluid.layers.reduce_mean(dice_loss + ce_loss)  # 最后使用的loss是dice和交叉熵的和


def focal_loss(predict, label, num_classes=num_classes):
    # 使用focal loss 效果不怎么好
    label = paddle.cast(label, dtype='int32')
    predict = paddle.transpose(predict, perm=[0, 2, 3, 1])  # shape[1, 512, 512, 2]
    predict = paddle.reshape(predict, shape=[-1, num_classes])  # [num, 2]
    one = paddle.to_tensor([1.], dtype='int32')
    fg_label = paddle.greater_equal(label, one)
    fg_num = paddle.sum(paddle.cast(fg_label, dtype='int32'))
    loss = fluid.layers.sigmoid_focal_loss(x=predict, label=label, fg_num=fg_num, gamma=2.0, alpha=0.3)
    return paddle.mean(loss)


def mean_iou(pred, label, num_classes=num_classes):
    ''' 计算miou，评价网络分割结果的指标'''
    pred = paddle.argmax(pred, axis=1)
    label = np.squeeze(label, axis=1)
    pred = paddle.cast(pred, 'int32')  # 转换数据类型
    label = paddle.cast(label, 'int32')
    miou, wrong, correct = paddle.fluid.layers.mean_iou(pred, label, num_classes)
    # miou, wrong, correct = paddle.metric.mean_iou(pred, label, num_classes)  # 计算均值IOU
    return miou


def train(model):
    print('开始训练 ... ')
    best_iou = 0.0
    model.train()
    scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.0002, decay_steps=5, end_lr=0.0000125,
                                                    cycle=True, verbose=True)

    opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

    train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = paddle.to_tensor(data[0], dtype='float32')
            y_data = paddle.to_tensor(data[1], dtype='int64')  # CrossEntropyLoss标签数据要int64格式   形状为 [N]
            y_data = paddle.reshape(y_data, (-1, 1))
            output = model(x_data)[0]
            save_data = np.expand_dims(x_data[0], axis=0)
            save_data = paddle.to_tensor(save_data, dtype='float32')
            # acc = mean_iou(output[0], y_data)
            # loss = create_loss(output[0], y_data, num_classes=num_classes)
            acc = mean_iou(output, y_data)
            loss = create_loss(output, y_data, num_classes=num_classes)

            if batch_id % 20 == 0:
                print("count: {}, epoch: {}, batch_id: {}, miou is :{} ,loss is: {}".format(count, epoch, batch_id,
                                                                                            acc.numpy(),
                                                                                            loss.numpy()))
            loss.backward()
            opt.minimize(loss)
            model.clear_gradients()
        scheduler.step()

        # 训练期间验证
        model.eval()
        meaniou = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = paddle.to_tensor(data[0], dtype='float32')
            y_data = paddle.to_tensor(data[1], dtype='int64')
            output = model(x_data)[0]
            acc = mean_iou(output, y_data)
            y_data = paddle.reshape(y_data, (-1, 1))
            # loss = create_loss(output[0], y_data, num_classes=num_classes)
            loss = create_loss(output, y_data, num_classes=num_classes)

            meaniou.append(np.mean(acc.numpy()))
            losses.append(np.mean(loss.numpy()))
        avg_iou, avg_loss = np.mean(meaniou), np.mean(losses)
        print("[validation] miou/loss: {}/{}".format(avg_iou, avg_loss))
        val_acc_history.append(avg_iou)
        val_loss_history.append(avg_loss)
        model.train()
        # paddle.onnx.export(model, 'c++_onnx.save'+ str(epoch) +'/unet' , input_spec=[save_data], opset_version=12, enable_onnx_checker=True)
        # print('成功保存onnx模型')
        if epoch % 10 == 0:
            paddle.save(model.state_dict(), "st_frame/" + "leaky_unet" + str(epoch) + "_net.pdparams")
            print('成功保存模型')
            paddle.onnx.export(model, './st_frame_onnx/leaky_unet_onnx.save' + str(epoch) + '/unet',
                               input_spec=[save_data], opset_version=12, enable_onnx_checker=True)
            print('成功保存onnx模型')
        if avg_iou > best_iou:
            best_iou = avg_iou
            paddle.save(model.state_dict(), "st_frame/" + "leaky_unet" + "best" + "_net.pdparams")
            print('成功最佳保存模型')
            # paddle.onnx.export(model, './onnx_out/attention_best_onnx.save'+ '/unet' , input_spec=[save_data], opset_version=12, enable_onnx_checker=True)
            # print('成功保存最佳onnx模型')


def predict(model_path):
    val_dataset = MyDataset(mode='val',
                            txt_file='work/train_list.txt')  # TODO 修改为读取train_list.txt中的所有origin后，预测并修改对应的label
    model_leaky = Leaky_UNet(num_classes=3)
    para_state_dict_leaky = paddle.load(model_path)  # TODO 就用上一轮训练的模型参数
    model_leaky.set_state_dict(para_state_dict_leaky)
    for k in range(len(val_dataset)):
        if random.random() > 0.3:
            imga, imgb = val_dataset[k]
            # line = val_dataset[k].lines[0]  # ('fine_data/test/50/origin\\0001.png', 'fine_data/test/50/label\\0001.png')
            save_path = val_dataset.lines[k][1]  # 'fine_data/test/50/label\\0001.png'
            imga = np.expand_dims(imga, axis=0)

            x_data = paddle.to_tensor(imga, dtype='float32')

            model_leaky.eval()
            output_leaky = model_leaky(x_data)[0]

            output_leaky = output_leaky.numpy()
            output_leaky = np.argmax(output_leaky, axis=1)
            output_leaky = output_leaky.transpose(1, 2, 0)

            cv2.imwrite(save_path, output_leaky)


model_path = 'fine_sam_out/leaky_unet100_net.pdparams'
for i in range(5):
    # # Teacher Prediction -> Change Labels(2) -> 更新label中的图片即可
    predict(model_path)
    print('数据集已更新')
    count += 1
    # TODO re-train

    # 第一次训练stu用之前的参数（目的为了保证标签的正确性）
    model = Leaky_UNet(num_classes=3, pretrained=model_path)
    train(model)
    model_path = 'st_frame/leaky_unet20_net.pdparams'
