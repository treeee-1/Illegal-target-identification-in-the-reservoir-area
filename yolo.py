
# import os
# import time
import colorsys
from PIL import ImageDraw, ImageFont

import cv2
import numpy as np
import torch
import torch.nn as nn
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, cvt_color_batch, resize_image_batch)
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {

        "model_path"        : 'logs/ep069-loss1.121-val_loss0.953.pth',
        "classes_path"      : 'model_data/boat.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.35,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # --------------------切割图片----------------------------------------------------------#
    def crop_one_picture(self, path, filename, cols, rows, overlap_cols, overlap_rows):
        '''对一张大图进行裁剪，记录裁剪后各小图在大图中的位置（左上角）'''
        big_img = cv2.imread(path + filename, 1)
        sum_rows = big_img.shape[0]  # 高度
        sum_cols = big_img.shape[1]  # 宽度
        # save_path =path + "crop{0}_{1}/".format(cols, rows)  # 保存的路径
        left_corner_positions = [] # 记录切割后的各个小图在原始大图中的位置
        all_images = []
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        for i, i_start in enumerate(range(0, sum_cols, cols - overlap_cols)):  # 宽
            for j, j_start in enumerate(range(0, sum_rows, rows - overlap_rows)):  # 高
                i_end = i_start + cols  # 重叠度 = cols - 步长
                j_end = j_start + rows
                img = big_img[j_start: j_end, i_start: i_end, :]
                left_corner_positions.append((j_start, i_start)) # 小图左上角坐标
                # filenames = save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + \
                #             os.path.splitext(filename)[1]
                if img.shape[0] == rows and img.shape[1] == cols:
                    all_images.append(img)
                    # cv2.imwrite(filenames, img)
                elif img.shape[0] == rows and img.shape[1] <= cols:
                    img = np.pad(img, ((0, 0), (0, cols - img.shape[1]), (0, 0)), 'constant', constant_values=0)
                    all_images.append(img)
                    # cv2.imwrite(filenames, img)
                elif img.shape[0] <= rows and img.shape[1] == cols:
                    img = np.pad(img, ((0, rows - img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
                    all_images.append(img)
                    # cv2.imwrite(filenames, img)
                elif img.shape[0] <= rows and img.shape[1] <= cols:
                    img = np.pad(img, ((0, rows - img.shape[0]), (0, cols - img.shape[1]), (0, 0)), 'constant',
                                 constant_values=0)
                    all_images.append(img)
                    # cv2.imwrite(filenames, img)
        return all_images, left_corner_positions
    #----------------检测图片-------------------#
    def detect_image_batch(self, images):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(images[0])[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        images = cvt_color_batch(images)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image_batch(images, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#

        image_data = np.array([np.array(image) for image in image_data], dtype='float32')
        image_data = np.transpose(preprocess_input(image_data), (0, 3, 1, 2))

        with torch.no_grad():
            input_images = torch.from_numpy(image_data)
            if self.cuda:
                input_images = input_images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(input_images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        image = images[0]
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        image_output = []
        predictions = []

        for i_image in range(len(images)):
            output = [outputs[i][i_image:i_image+1] for i in range(len(outputs))]

            result = self.bbox_util.non_max_suppression(torch.cat(output, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
            # print(result)
            # if results[0] is None:
            #     return image
            image = images[i_image]

            # ---------------------------------------------------------#
            #   图像绘制
            # ---------------------------------------------------------#
            result = result[0] #非极大值抑制之后的output，只剩一个
            if result is None:
                image_output.append(image)
                predictions.append(None)
                continue
            top_labels = np.array(result[:, 6], dtype='int32') #6是类别标签的索引，只有一类：0
            top_conf = result[:, 4] * result[:, 5] #result数组第四个位置和第五个位置相乘，得到置信度
            top_boxes = result[:, :4] #矩阵的索引，取前四个
            image_after_draw = self.draw_image(image, top_boxes, top_labels, top_conf, font, thickness)

            image_output.append(image_after_draw)
            predictions.append((top_boxes, top_labels, top_conf))

        return image_output, predictions

    def draw_image(self, image, boxs, top_classes, top_conf, font, thickness):
        '''保存单个图片的绘制结果'''
        draw = ImageDraw.Draw(image)
        for box, c, conf in zip(boxs, top_classes, top_conf):
            c = int(c)
            label = '{} {:.2f}'.format(self.class_names[c], conf)
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            # corner_x = (right - left) / 2 + left
            # print(label, top, left, bottom, right,corner_x)

            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

        #del draw

        return image

