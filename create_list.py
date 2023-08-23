# 创建train.txt, val.txt, test.txt 文档
import os
import random

random.seed(2021)

train_path = 'data/train/origin'
img_path_list = list()
index = 0
#train.txt
with open('data/train_list.txt', 'w') as f:
    for folder in os.listdir(train_path):
        for img in os.listdir(os.path.join(train_path,folder)):
            img_path = os.path.join(train_path,folder,img)
            content = img_path  + '  ' + img_path.replace('origin','label') + '\n'
            img_path_list.append(content)
            index += 1
    # shuffle(img_path_list)
    img_path_list.sort()
    for path in img_path_list:
        f.write(path)

#train_txt = open('work/train_list.txt','w')
val_txt = open('data/val_list.txt','w')
test_txt = open('data/test_list.txt','w')

train_path_origin = 'data/train/origin'
train_path_label = 'data/train/label'
files = list(filter(lambda x: x.endswith('.png'), os.listdir(train_path_origin)))

random.shuffle(files)


val_path_origin = 'data/val/1/origin'
val_path_label = 'data/val/1/label'
files = list(filter(lambda x: x.endswith('.png'), os.listdir(val_path_origin)))
for i,f in enumerate(files):
    val_image_path = os.path.join(val_path_origin, f)
    val_label_name = f.split('.')[0]+ '.png'
    val_label_path = os.path.join(val_path_label, val_label_name)
    val_txt.write(val_image_path + ' ' + val_label_path+ '\n')
val_txt.close()

test_path_origin = 'data/test/61/origin'
test_path_label = 'data/test/61/label'
files = list(filter(lambda x: x.endswith('.png'), os.listdir(test_path_origin)))
for i,f in enumerate(files):
    test_image_path = os.path.join(test_path_origin, f)
    test_label_name = f.split('.')[0]+ '.png'
    test_label_path = os.path.join(test_path_label, test_label_name)
    test_txt.write(test_image_path + ' ' + test_label_path+ '\n')
test_txt.close()

print('数据列表创建完成')