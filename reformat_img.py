
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL

def reformat_img():

    # path
    train_labels_path_s = "D:/IP5/data/labels/small/train.txt"
    test_labels_path_s = "D:/IP5/data/labels/small/test.txt"
    val_labels_path_s = "D:/IP5/data/labels/small/val.txt"

    dataset_dir = "D:/IP5/data/images/"
    dest_dir = "D:/IP5/data/1000x800/png/"

    max_height = 1000
    max_width = 800

    # shape(max_height, max_width, rgb)
    # extract train images and labels
    file = open(test_labels_path_s, 'r')
    for line in file:
        index = line.find(" ")
        src = line[0:index]
        src = dataset_dir + src
        label = int(line[index + 1:-1])
        dst = ""
        flag = -1
        for i in range(len(src)):
            if src[i] != '/':
                continue
            flag = i
        image = np.zeros((max_height, max_width), np.float16)
        im = PIL.Image.open(src)
        arr = np.array(im, order='F')
        border = arr.shape[1]
        if border > max_width:
            border = max_width
            arr.resize(max_height, border)
        # add greyscale image to data set
        image[0:max_height, 0:border] += arr
        data = PIL.Image.fromarray(image.astype('uint8'))
        if label == 0:
            dst = dest_dir + "0"
        elif label == 1:
            dst = dest_dir + "1"
        elif label == 2:
            dst = dest_dir + "2"
        elif label == 3:
            dst = dest_dir + "3"
        elif label == 4:
            dst = dest_dir + "4"
        else:
            print("No label given")
        dst = dst + src[flag:-3] + "png"
        data.save(dst)

    file.close()
