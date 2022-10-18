import json
import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvt_color_batch(images):
    if len(np.shape(images[0])) == 3 and np.shape(images[0])[-1] == 3:
        return images
    else:
        images_new = []
        for image in images:
            images_new.append(image.convert('RGB'))
        return images_new

def prediction_2_json(pred_results, save_file):
    boxes, labels, confs = pred_results
    with open(save_file, 'w', encoding='utf-8') as writer:
        for i, (box, label, conf) in enumerate(zip(boxes, labels, confs)):
            json_d = json.dumps(
                {
                    'id': float(i),

                    #'box': [box[4],box[5]],
                    'box': [float(x) for x in box],
                    'label': float(label.item()),
                    'conf': conf.item()
                }, ensure_ascii=False
            )

            writer.write(json_d)
            writer.write('\n')



def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#

def resize_image_batch(images, size, letterbox_image):

    iw, ih = images[0].size
    w, h = size
    new_images = []
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        for image in images:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            new_images.append(new_image)
    else:
        for image in images:
            new_image = image.resize((w, h), Image.BICUBIC)
            new_images.append(new_image)
    return new_images

def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image