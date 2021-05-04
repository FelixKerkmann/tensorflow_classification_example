# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_io as tfio

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

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]),
        color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def clothing():
    print(tf.__version__)

    #fashion_mnist = tf.keras.datasets.fashion_mnist

    #(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    dataset_dir = "D:/ip5_dataset/images/"

    test_labels_path = "D:/ip5_dataset/labels/test.txt"
    train_labels_path = "D:/ip5_dataset/labels/train.txt"
    val_labels_path = "D:/ip5_dataset/labels/val.txt"
    
    n = 100
    m = 1000
    
    train_images = np.arange(n*m*m).reshape(n, m, m, 1)
    train_labels = np.arange(n)
    test_images = np.arange(n*2*m*m).reshape(n*2, m, m, 1)
    test_labels = np.arange(n*2)
    val_images = np.arange(n*2*m*m).reshape(n*2, m, m, 1)
    val_labels = np.arange(n*2)
    
    train_file = open(train_labels_path, 'r')
    lines = train_file.readlines()
    for i in range(n):
        line = lines[i]
        index = line.find(" ")
        img = tf.io.read_file(dataset_dir + line[0:index])
        decoded = tfio.experimental.image.decode_tiff(img)
        rgb = tfio.experimental.color.rgba_to_rgb(decoded)
        grayscale = tf.image.rgb_to_grayscale(rgb)
        dim = grayscale.shape
        #grayscale image array in form (m, m, 1) bringen (mit 0 füllen)
        if m > dim[0]:
            grayscale = np.hstack([grayscale, np.zeros([(m-dim[0]), dim[1], 1])])
        if m > dim[1]:
            grayscale = np.hstack([grayscale, np.zeros([dim[0], (m-dim[1]), 1])])
        train_images[i,:] = grayscale
        train_labels[i] = line[index + 1:-1]
    train_file.close()
    
    print(train_images)
    
    test_file = open(test_labels_path, 'r')
    lines = test_file.readlines()
    for i in range(2*n):
        line = lines[i]
        index = line.find(" ")
        img = tf.io.read_file(dataset_dir + line[0:index])
        decoded = tfio.experimental.image.decode_tiff(img)
        rgb = tfio.experimental.color.rgba_to_rgb(decoded)
        grayscale = tf.image.rgb_to_grayscale(rgb)
        dim = grayscale.shape
        #grayscale image array in form (m, m, 1) bringen (mit 0 füllen)
        if m > dim[0]:
            grayscale = np.hstack([grayscale, np.zeros([(m-dim[0]), dim[1], 1])])
        if m > dim[1]:
            grayscale = np.hstack([grayscale, np.zeros([dim[0], (m-dim[1]), 1])])
        test_images[i,:] = grayscale
        test_labels[i] = line[index + 1:-1]
    test_file.close()
    
    val_file = open(test_labels_path, 'r')
    lines = val_file.readlines()
    for i in range(2*n):
        line = lines[i]
        index = line.find(" ")
        img = tf.io.read_file(dataset_dir + line[0:index])
        decoded = tfio.experimental.image.decode_tiff(img)
        rgb = tfio.experimental.color.rgba_to_rgb(decoded)
        grayscale = tf.image.rgb_to_grayscale(rgb)
        dim = grayscale.shape
        #grayscale image array in form (m, m, 1) bringen (mit 0 füllen)
        if m > dim[0]:
            grayscale = np.hstack([grayscale, np.zeros([(m-dim[0]), dim[1], 1])])
        if m > dim[1]:
            grayscale = np.hstack([grayscale, np.zeros([dim[0], (m-dim[1]), 1])])
        val_images[i,:] = grayscale
        val_labels[i] = line[index + 1:-1]
    val_file.close()
    
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)

    print(test_images.shape)
    print(len(test_labels))

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(1000, 1000, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=16)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])

    #i = 0
    #plt.figure(figsize=(6, 3))
    #plt.subplot(1, 2, 1)
    #plot_image(i, predictions[i], test_labels, test_images)
    #plt.subplot(1, 2, 2)
    #plot_value_array(i, predictions[i], test_labels)
    #plt.show()

    #i = 12
    #plt.figure(figsize=(6, 3))
    #plt.subplot(1, 2, 1)
    #plot_image(i, predictions[i], test_labels, test_images)
    #plt.subplot(1, 2, 2)
    #plot_value_array(i, predictions[i], test_labels)
    #plt.show()

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    #num_rows = 5
    #num_cols = 3
    #num_images = num_rows * num_cols
    #plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    #for i in range(num_images):
        #plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        #plot_image(i, predictions[i], test_labels, test_images)
        #plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        #plot_value_array(i, predictions[i], test_labels)
    #plt.tight_layout()
    #plt.show()

    # Grab an image from the test dataset.
    #img = test_images[1]
    #print(img.shape)

    # Add the image to a batch where it's the only member.
    #img = (np.expand_dims(img, 0))
    #print(img.shape)

    #predictions_single = probability_model.predict(img)
    #print(predictions_single)

    #plot_value_array(1, predictions_single[0], test_labels)
    #_ = plt.xticks(range(10), class_names, rotation=45)
    #np.argmax(predictions_single[0])

clothing()