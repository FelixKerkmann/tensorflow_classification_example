import shutil
import os
import openpyxl
import pdf2image
import numpy as np
import PIL

if __name__ == '__main__':

    max_height = 1000
    max_width = 800

    src_dir = 'D:/IP5/USB/resized/'
    dst_dir = 'D:/IP5/USB/data/'

    cat_list = os.listdir(src_dir)
    for folder in cat_list:
        file_list = os.listdir(src_dir + folder + '/')
        print("folder: ", folder)
        for img in file_list:
            src = src_dir + folder + '/' + img
            dst = dst_dir + folder + '/' + img
            print("src: ", src)
            print("dst: ", dst)
            image = np.zeros((max_height, max_width, 3), np.int8)
            im = PIL.Image.open(src)
            arr = np.array(im, order='F')
            print(arr.shape)
            border = arr.shape[1]
            if border > max_width:
                border = max_width
                arr.resize(max_height, border, 3)
            # add greyscale image to data set
            image[0:max_height, 0:border, :] += arr
            data = PIL.Image.fromarray(image.astype('uint8'))
            data.save(dst)
