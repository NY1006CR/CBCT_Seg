import paddle
import paddle.fluid as fluid
from paddleseg.models import UNet
import numpy as np

from cbct_seg.Dataset.dataset import MyDataset

num_classes = 3
val_acc_history = []
val_loss_history = []
epoch_num = 100
batch_size = 2
learning_rate = 0.002

train_dataset = MyDataset(mode='train', txt_file='data/train_list.txt')
val_dataset = MyDataset(mode='val', txt_file='data/val_list.txt')


def create_loss(predict, label, num_classes=num_classes):
    ''' 创建loss，结合dice和交叉熵 '''
    predict = paddle.transpose(predict, perm=[0, 2, 3, 1])  # shape[1, 512, 512, 2]
    predict = paddle.reshape(predict, shape=[-1, num_classes])  # [num, 2]
    m = paddle.nn.Softmax()
    predict = m(predict)
    ce_loss = paddle.nn.loss.CrossEntropyLoss(ignore_index=0, reduction='mean')
    ce_loss = ce_loss(predict, label)  # 计算交叉熵
    dice_loss = fluid.layers.dice_loss(predict, label)  # 计算dice loss
    return fluid.layers.reduce_mean(dice_loss + ce_loss)



def mean_iou(pred, label, num_classes=num_classes):
    pred = paddle.argmax(pred, axis=1)
    label = np.squeeze(label, axis=1)
    pred = paddle.cast(pred, 'int32')
    label = paddle.cast(label, 'int32')
    miou, wrong, correct = paddle.fluid.layers.mean_iou(pred, label, num_classes)
    return miou


def train(model):
    print('开始训练 ... ')
    best_iou = 0.0
    model.train()
    scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=learning_rate, decay_steps=20, end_lr=0.0000125,
                                                    cycle=True, verbose=True)
    # scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.000005, decay_steps=20, end_lr=0.00000125,
    #                                                 cycle=True, verbose=True)
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
            acc = mean_iou(output, y_data)
            loss = create_loss(output, y_data, num_classes=num_classes)

            if batch_id % 10 == 0:
                print("epoch: {}, batch_id: {}, miou is :{} ,loss is: {}".format(epoch, batch_id, acc.numpy(),
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
            loss = create_loss(output, y_data, num_classes=num_classes)

            meaniou.append(np.mean(acc.numpy()))
            losses.append(np.mean(loss.numpy()))
        avg_iou, avg_loss = np.mean(meaniou), np.mean(losses)
        print("[validation] miou/loss: {}/{}".format(avg_iou, avg_loss))
        val_acc_history.append(avg_iou)
        val_loss_history.append(avg_loss)
        model.train()
        if epoch % 50 == 0:
            paddle.save(model.state_dict(), "out/UNet/100/" + "unet" + str(epoch) + ".pdparams")
            print('成功保存模型')
            paddle.onnx.export(model, 'onnx.save' + str(epoch) + '/unet', input_spec=[save_data], opset_version=12,
                               enable_onnx_checker=True)
            print('成功保存onnx模型')
        if avg_iou > best_iou:
            best_iou = avg_iou
            paddle.save(model.state_dict(), "out/UNet/100/" + "unet" + "best" + "_net.pdparams")
            print('成功最佳保存模型')

model = UNet(num_classes=num_classes,
             pretrained=None)
train(model)