# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL

if __name__ == '__main__':
    # load model
    new_model = tf.keras.models.load_model('saved_model/my_model')

    # Check its architecture
    new_model.summary()

    # evaluate
    loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    # predictions
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
