import shutil
import os

if __name__ == '__main__':

    train_labels_path_s = "D:/IP5/data/labels/small/train.txt"
    test_labels_path_s = "D:/IP5/data/labels/small/test.txt"
    val_labels_path_s = "D:/IP5/data/labels/small/val.txt"

    dest_dir = "D:/IP5/data/categories/"
    dataset_dir = "D:/IP5/data/images/"

    # extract every class and copy image to one categorized directory

    val_file = open(val_labels_path_s, 'r')
    for line in val_file:
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
        if label == 0:
            dst = dest_dir + "0" + src[flag:]
        elif label == 1:
            dst = dest_dir + "1" + src[flag:]
        elif label == 2:
            dst = dest_dir + "2" + src[flag:]
        elif label == 3:
            dst = dest_dir + "3" + src[flag:]
        elif label == 4:
            dst = dest_dir + "4" + src[flag:]
        else:
            print("No label given")
        # shutil.copy(src, dst)
        print(src, dst)




