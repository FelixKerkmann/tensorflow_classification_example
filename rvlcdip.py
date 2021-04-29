# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL

class_names = ['letter',
               'form',
               'email',
               'handwritten',
               'advertisement',
               'scientific report',
               'scientific publication',
               'specification',
               'file folder',
               'news article',
               'budget',
               'invoice',
               'presentation',
               'questionnaire',
               'resume',
               'memo']


def rvlcdip():
    dataset_dir = "D:/IP5/data/images/"

    test_labels_path = "D:/IP5/data/labels/test.txt"
    train_labels_path = "D:/IP5/data/labels/train.txt"
    val_labels_path = "D:/IP5/data/labels/val.txt"

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    val_images = []
    val_labels = []

    # extract train images and labels
    train_file = open(train_labels_path, 'r')
    for line in train_file:
        index = line.find(" ")
        img_path = line[0:index]
        label = line[index + 1:-1]
        train_images.append(img_path)
        train_labels.append(label)
    train_file.close()

    # extract test images and labels
    test_file = open(test_labels_path, 'r')
    for line in test_file:
        index = line.find(" ")
        img_path = line[0:index]
        label = line[index+1:-1]
        test_images.append(img_path)
        test_labels.append(label)
    test_file.close()

    # extract validation images and labels
    val_file = open(val_labels_path, 'r')
    for line in val_file:
        index = line.find(" ")
        img_path = line[0:index]
        label = line[index + 1:-1]
        val_images.append(img_path)
        val_labels.append(label)
    val_file.close()

    # print(test_images)
    # print(test_labels)

    im = PIL.Image.open(str(dataset_dir+test_images[0]))
    im.show()
