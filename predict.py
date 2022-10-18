# coding=utf-8
from osgeo import gdal
from gdalconst import *
import time
import cv2
import numpy as np
import os
from yolo import YOLO
from PIL import Image, ImageFont
import sys
if __name__ == "__main__":
    yolo = YOLO()
    mode = "dir_predict_batch"

    #------------------------------------------------------------------------#
    #   dir_save_path指定了检测完图片的保存路径
    #-------------------------------------------------------------------------#

    # path = input('输入要裁剪的图片所在的文件夹路径比如"F:/tifs/"')
    # filename = input('输入要裁剪的图片名比如"a.tif"')
    path = sys.argv[1]
    filename = sys.argv[2]
    file_name = filename.split('.')[0]
    if not os.path.exists(path + file_name):
        os.mkdir(path + file_name)

    dir_save_path = path + file_name

    cols = 880  # 小图片的宽度（列数）
    rows = 650  # 小图片的高度（行数）
    overlap_cols = 60
    overlap_rows = 60
    # 裁剪并返回各小图在大图中的位置
    all_images, left_corner_positions = yolo.crop_one_picture(path, filename, cols, rows, overlap_cols, overlap_rows)
    # dir_save_path   = "result_out/"

    # start_time = time.time()

    if mode == "dir_predict_batch":
        images = []
        for img in all_images:
            images.append(Image.fromarray(img))
        img_names = [f'crop_{i}.tif' for i in range(len(images))]
        r_images = []
        all_predictions = []
        end = 0
        bs = 16
        bs_index = 0
        for end in range(bs, len(img_names), bs):
            input_images = images[end-bs: end]
            img = cv2.imread(path + filename, 1)
            sum_rows = img.shape[0]  # 高度
            sum_cols = img.shape[1]
            r_image, predictions = yolo.detect_image_batch(input_images)
            r_images.extend(r_image)
            all_predictions.extend(predictions)

        if end <= len(img_names) - 1:
            input_images = images[end: len(img_names)]
            r_image, predictions = yolo.detect_image_batch(input_images)
            r_images.extend(r_image)  #检测完一张图像上的所有目标，然后再extend。所以检测时，对每张图片的所有预测框进行坐标转换完成后，再处理下一张
            all_predictions.extend(predictions)
        # if not os.path.exists(dir_save_path):
        #     os.makedirs(dir_save_path)
        # for i, r_image in enumerate(r_images):
        #     r_image.save(os.path.join(dir_save_path, img_names[i]))

        #gdal库读取tif坐标
        gdalimg_path = os.path.join(path,filename)
        dataset = gdal.Open(gdalimg_path, GA_ReadOnly)
        geoTransform = dataset.GetGeoTransform()

        # print(geoTransform[0], geoTransform[3], geoTransform[1])

        # def get_global_boxes(local_boxes, local_index):
        #     global_boxes = None
        #     return global_boxes

        assert len(left_corner_positions) == len(all_predictions)
        all_boxes = []
        all_labels = []
        all_conf = []
        min_all_boxes = []
        for i, (pred_result, left_corner) in enumerate(zip(all_predictions, left_corner_positions)):
            if pred_result is None: continue
            local_boxes, local_labels, local_conf = pred_result
            left_row, left_col = left_corner
            image = r_images[i]

            for box in local_boxes:
                top, left, bottom, right = box
                if top < 0 or left < 0 or bottom < 0 or right < 0:
                    print('stop here.')
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                min_global_top = left_row + top
                min_global_left = left_col + left
                min_global_bottom = left_row + bottom
                min_global_right = left_col + right
                min_global_box = (min_global_top, min_global_left, min_global_bottom, min_global_right)
                min_all_boxes.append(min_global_box)

                global_left = geoTransform[0] + (left_col + left) * geoTransform[1]
                global_top = geoTransform[3] + (left_row  + top) * geoTransform[5]

                global_right = geoTransform[0] + (left_col + right) * geoTransform[1]
                global_bottom = geoTransform[3] + (left_row + bottom) * geoTransform[5]

                cornerx = (global_right - global_left) * 0.5 + global_left
                cornery = (global_bottom - global_top) * 0.5 + global_top
                global_box = (cornerx, cornery)

                all_boxes.append(global_box)

            all_labels.extend(local_labels)
            all_conf.extend(local_conf)
        print(all_boxes)
        print(len(all_boxes))
        # prediction_2_json((all_boxes, all_labels, all_conf), save_file=os.path.join(dir_save_path, 'prediction.jl'))
        #print('saved.')
        # # verify
        big_img = Image.open(os.path.join(path, filename))
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(5e-3 * big_img.size[1] + 0.5).astype('int32'))
        #将目标框绘制到到大图上，并保存
        big_img_res = yolo.draw_image(big_img, min_all_boxes, all_labels, all_conf, font, thickness=1)

        big_img_res.save(os.path.join(dir_save_path, 'big_image_after_draw.png'))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

    # end_time = time.time()
    #
    # print(f'used time: {end_time-start_time}')


