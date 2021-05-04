# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_io as tfio

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import pathlib

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


def getmaxshape(path):
    file = open(path, 'r')
    maxshape = [0, 0]
    nrofelem = 0
    brokenfiles = []
    for line in file:
        index = line.find(" ")
        img_path = line[0:index]
        try:
            im = PIL.Image.open(str('D:/IP5/data/images/{0}'.format(img_path))).convert('L')
            greyscale_map = np.array(im)
            if greyscale_map.shape[0] <= maxshape[0]:
                pass
            else:
                maxshape[0] = greyscale_map.shape[0]
                print(line, maxshape)
            if greyscale_map.shape[1] <= maxshape[1]:
                pass
            else:
                maxshape[1] = greyscale_map.shape[1]
                print(line, maxshape)
            im.close()
        except PIL.UnidentifiedImageError:
            brokenfiles.append(line)
        nrofelem = nrofelem + 1
    file.close()

    return maxshape, nrofelem, brokenfiles


def rvlcdip():

    dataset_dir = "D:/IP5/data/images/"

    test_labels_path = "D:/IP5/data/labels/test.txt"
    train_labels_path = "D:/IP5/data/labels/train.txt"
    val_labels_path = "D:/IP5/data/labels/val.txt"

    '''
    (test_lim, test_nr, test_brokenfiles) = getmaxshape(test_labels_path)
    print(test_lim) => [1000, 2542]
    print(test_nr) => 40000
    print(test_brokenfiles) => ['imagese/e/j/e/eje42e00/2500126531_2500126536.tif 6'] (nr 328)
    '''
    '''
    (train_lim, train_nr, train_brokenfiles) = getmaxshape(train_labels_path)
    print(train_lim) => [1000, 3235]
    print(train_nr) => 320000
    print(train_brokenfiles) => []
    '''
    '''
    (val_lim, val_nr, val_brokenfiles) = getmaxshape(val_labels_path)
    print(val_lim) => [1000, 2544]
    print(val_nr) => 40000
    print(val_brokenfiles) => []
    '''
    '''
    nr_train = 320000
    nr_test = 39999
    nr_val = 40000
    '''
    nr_train = 4000
    nr_test = 1000
    nr_val = 1000
    max_width = 1000
    max_height = 1000

    train_images = np.zeros((nr_train, max_height, max_width), np.float16)
    train_labels = np.zeros(nr_train)
    test_images = np.zeros((nr_test, max_height, max_width), np.float16)
    test_labels = np.zeros(nr_test)
    val_images = np.zeros((nr_val, max_height, max_width), np.float16)
    val_labels = np.zeros(nr_val)
    # print(test_images)
    # print(test_labels)

    # extract train images and labels
    train_file = open(train_labels_path, 'r')
    lines = train_file.readlines()
    # for line in train_file:
    for i in range(nr_train):
        line = lines[i]
        index = line.find(" ")
        im = PIL.Image.open(str(dataset_dir + line[0:index]))
        im_conv = im.convert('L')
        greyscale_map = np.array(im_conv, order='F')
        border = greyscale_map.shape[1]
        if border > max_width:
            border = max_width
            greyscale_map.resize(max_height, border)
        train_images[i, 0:max_height, 0:border] += greyscale_map
        im.close()
        train_labels[i] = line[index + 1:-1]
    train_file.close()
    # print(train_images)
    # print(train_labels)

    # extract test images and labels
    test_file = open(test_labels_path, 'r')
    lines = test_file.readlines()
    # for line in test_file:
    for i in range(nr_test):
        line = lines[i]
        index = line.find(" ")
        im = PIL.Image.open(str(dataset_dir + line[0:index]))
        im_conv = im.convert('L')
        greyscale_map = np.array(im_conv, order='F')
        border = greyscale_map.shape[1]
        if border > max_width:
            border = max_width
            greyscale_map.resize(max_height, border)
        test_images[i, 0:max_height, 0:border] += greyscale_map
        im.close()
        test_labels[i] = line[index + 1:-1]
    test_file.close()
    # print(test_images)
    # print(test_labels)

    # extract val images and labels
    val_file = open(val_labels_path, 'r')
    lines = val_file.readlines()
    # for line in val_file:
    for i in range(nr_val):
        line = lines[i]
        index = line.find(" ")
        im = PIL.Image.open(str(dataset_dir + line[0:index]))
        im_conv = im.convert('L')
        greyscale_map = np.array(im_conv, order='F')
        border = greyscale_map.shape[1]
        if border > max_width:
            border = max_width
            greyscale_map.resize(max_height, border)
        val_images[i, 0:max_height, 0:border] += greyscale_map
        im.close()
        val_labels[i] = line[index + 1:-1]
    val_file.close()
    # print(val_images)
    # print(val_labels)

    ''' 
    # Control
    print(train_images.shape)
    print(test_images.shape)
    print(val_images.shape)
    print(train_labels.shape)
    print(test_labels.shape)
    print(val_labels.shape)
    '''

    # Normalize
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(max_height, max_width)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=len(class_names))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    val_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    validations = val_model.predict(val_images)

    print(validations[0])
    print(np.argmax(validations[0]))
    print(val_labels[0])




