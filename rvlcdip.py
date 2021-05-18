# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import pathlib

# import the models for further classification experiments
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# all 16 classes
'''
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
'''

# only 5 classes
class_names = ['email',
               'handwritten',
               'advertisement',
               'scientific report',
               'scientific publication']

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
    data_dir = "D:/IP5/data/1000x800/png"
    model_dir = "D:/IP5/saved_model/"
    # path 16 classes
    '''
    test_labels_path = "D:/IP5/data/labels/test.txt"
    train_labels_path = "D:/IP5/data/labels/train.txt"
    val_labels_path = "D:/IP5/data/labels/val.txt"
    '''

    # path 5 calsses
    test_labels_path = "D:/IP5/data/labels/small/test.txt"
    train_labels_path = "D:/IP5/data/labels/small/train.txt"
    val_labels_path = "D:/IP5/data/labels/small/val.txt"

    # getmaxshape test
    '''
    (test_lim, test_nr, test_brokenfiles) = getmaxshape(test_labels_path)
    print(test_lim) => [1000, 2542]
    print(test_nr) => 40000
    print(test_brokenfiles) => ['imagese/e/j/e/eje42e00/2500126531_2500126536.tif 6'] (nr 328)
    
    (train_lim, train_nr, train_brokenfiles) = getmaxshape(train_labels_path)
    print(train_lim) => [1000, 3235]
    print(train_nr) => 320000
    print(train_brokenfiles) => []
    
    (val_lim, val_nr, val_brokenfiles) = getmaxshape(val_labels_path)
    print(val_lim) => [1000, 2544]
    print(val_nr) => 40000
    print(val_brokenfiles) => []
    '''

    # nr of elements from  all 16 classes
    '''
    nr_train = 320000
    nr_test = 39999
    nr_val = 40000
    '''

    nr_train = 1000
    nr_test = 200
    nr_val = 200

    # image attributes
    max_height = 1000
    max_width = 800

    batch_size = 32

    # epochs
    epochs = 2

    # create dataset manually
    '''
    # create np array
    train_images = np.zeros((nr_train, max_height, max_width, 3), np.float16)
    train_labels = np.zeros(nr_train)
    test_images = np.zeros((nr_test, max_height, max_width, 3), np.float16)
    test_labels = np.zeros(nr_test)
    val_images = np.zeros((nr_val, max_height, max_width, 3), np.float16)
    val_labels = np.zeros(nr_val)
    # print(test_images)
    # print(test_labels)

    # shape(max_height, max_width, rgb)
    # extract train images and labels
    train_file = open(train_labels_path, 'r')
    lines = train_file.readlines()
    # for line in train_file:
    for i in range(nr_train):
        line = lines[i]
        index = line.find(" ")
        # open image in greyscale
        
        # im = PIL.Image.open(str(dataset_dir + line[0:index]))
        # im_conv = im.convert('L')
        # greyscale_map = np.array(im_conv, order='F')
        
        img = tf.io.read_file(dataset_dir + line[0:index])
        decoded = tfio.experimental.image.decode_tiff(img)
        rgb = tfio.experimental.color.rgba_to_rgb(decoded)
        p = np.array(rgb, order='F')
        border = p.shape[1]
        if border > max_width:
            border = max_width
            p.resize(max_height, border, 3)
        # add greyscale image to data set
        train_images[i, 0:max_height, 0:border, :] += p
        train_labels[i] = line[index + 1:-1]
    train_file.close()

    # extract test images and labels
    test_file = open(test_labels_path, 'r')
    lines = test_file.readlines()
    # for line in test_file:
    for i in range(nr_test):
        line = lines[i]
        index = line.find(" ")
        # open image in greyscale
        img = tf.io.read_file(dataset_dir + line[0:index])
        decoded = tfio.experimental.image.decode_tiff(img)
        rgb = tfio.experimental.color.rgba_to_rgb(decoded)
        p = np.array(rgb, order='F')
        border = p.shape[1]
        if border > max_width:
            border = max_width
            p.resize(max_height, border, 3)
        # add greyscale image to data set
        test_images[i, 0:max_height, 0:border, :] += p
        test_labels[i] = line[index + 1:-1]
    test_file.close()

    # extract val images and labels
    val_file = open(val_labels_path, 'r')
    lines = val_file.readlines()
    # for line in val_file:
    for i in range(nr_val):
        line = lines[i]
        index = line.find(" ")
        # open image in greyscale
        img = tf.io.read_file(dataset_dir + line[0:index])
        decoded = tfio.experimental.image.decode_tiff(img)
        rgb = tfio.experimental.color.rgba_to_rgb(decoded)
        p = np.array(rgb, order='F')
        border = p.shape[1]
        if border > max_width:
            border = max_width
            p.resize(max_height, border, 3)
        # add greyscale image to data set
        val_images[i, 0:max_height, 0:border, :] += p
        val_labels[i] = line[index + 1:-1]
    val_file.close()
    '''

    # shape (max_height, max_width)
    '''
    # extract train images and labels
    train_file = open(train_labels_path, 'r')
    lines = train_file.readlines()
    # for line in train_file:
    for i in range(nr_train):
        line = lines[i]
        index = line.find(" ")
        # open image in greyscale
        im = PIL.Image.open(str(dataset_dir + line[0:index]))
        im_conv = im.convert('L')
        greyscale_map = np.array(im_conv, order='F')
        border = greyscale_map.shape[1]
        if border > max_width:
            border = max_width
            greyscale_map.resize(max_height, border)
        # add greyscale image to data set
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
    '''
    print(train_images.shape)
    print(test_images.shape)
    print(val_images.shape)
    print(train_labels.shape)
    print(test_labels.shape)
    print(val_labels.shape)
    '''

    # Normalize
    '''
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0
    '''

    # pretrained models
    '''
    vgg_model = vgg16.VGG16(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    resnet_model = resnet50.ResNet50(weights='imagenet')
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')
    '''

    # create dataset automatically
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(max_height, max_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(max_height, max_width),
        batch_size=batch_size,
    )
    # create model
    model = keras.models.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(max_height, max_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names))
    ])

    # define optimizer, loss and metrics
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # save for previous use
    model.save('D:/IP5/saved_model/my_model')

    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    # print('Test accuracy:', test_acc)
    # print('Test loss:', test_loss)

    img = keras.preprocessing.image.load_img(
        "test.png", target_size=(max_height, max_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    # validation for manually imported dataset
    '''
    val_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    validations = val_model.predict(val_images)

    print(validations[0])
    print(np.argmax(validations[0]))
    print(val_labels[0])
    '''
