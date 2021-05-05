# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_io as tfio

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL

class_names = ['email',
               'handwritten',
               'advertisement',
               'scientific report',
               'scientific publication']

def smaller_set():

    train_labels_path = "D:/IP5/data/labels/train.txt"
    test_labels_path = "D:/IP5/data/labels/test.txt"
    val_labels_path = "D:/IP5/data/labels/val.txt"

    train_labels_path_s = "D:/IP5/data/labels/small/train.txt"
    test_labels_path_s = "D:/IP5/data/labels/small/test.txt"
    val_labels_path_s = "D:/IP5/data/labels/small/val.txt"

    # Extract 5 classes out of train set
    # write it in new file
    train_file = open(train_labels_path, 'r')
    train_write = open(train_labels_path_s, 'w+')
    train_write.truncate()

    for line in train_file:
        index = line.find(" ")
        img_path = line[0:index]
        label = int(line[index + 1:-1])
        if 2 <= label <= 6:
            train_write.write(img_path + " " + str(label-2) + "\n")

    train_file.close()
    train_write.close()

    # Extract 5 classes out of train set
    # write it in new file
    test_file = open(test_labels_path, 'r')
    test_write = open(test_labels_path_s, 'w+')
    test_write.truncate()

    for line in test_file:
        index = line.find(" ")
        img_path = line[0:index]
        label = int(line[index + 1:-1])
        if 2 <= label <= 6:
            test_write.write(img_path + " " + str(label - 2) + "\n")

    test_file.close()
    test_write.close()

    # Extract 5 classes out of train set
    # write it in new file
    val_file = open(val_labels_path, 'r')
    val_write = open(val_labels_path_s, 'w+')
    val_write.truncate()

    for line in val_file:
        index = line.find(" ")
        img_path = line[0:index]
        label = int(line[index + 1:-1])
        if 2 <= label <= 6:
            val_write.write(img_path + " " + str(label - 2) + "\n")

    val_file.close()
    val_write.close()
