import random
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pickle_in = open("y_train.pickle", "rb")
y_train = pickle.load(pickle_in)
pickle_in = open("X_train.pickle", "rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_val.pickle", "rb")
y_val = pickle.load(pickle_in)
pickle_in = open("X_val.pickle", "rb")
X_val = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y_test = pickle.load(pickle_in)
pickle_in = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_in)

batch_size = 32
categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
model = tf.keras.models.load_model('ANNv2.model')


def prepare(filepath):
    pic_size = 48
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (pic_size, pic_size))
    plt.imshow(new_array, cmap='gray')
    # plt.show()
    return new_array.reshape(-1, pic_size, pic_size, 1)


# losowanie obrazu do testu
test_path = ('database/test/')
test_folders = os.listdir(test_path)
print(test_folders)
selected_folder = random.choice(test_folders)
print(selected_folder)
path = ('database/test/'+selected_folder)
print(path)

files = os.listdir(path)
print(len(files))

random_filename = random.choice([x for x in os.listdir(path)
                                 if os.path.isfile(os.path.join(path, x))])
print(random_filename)

img = (path+"/"+random_filename)
image = cv2.imread(img, 0)
# cv2.imshow("Obraz1", image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
train_results = model.evaluate(
    X_train, y_train, batch_size=batch_size)
print(train_results)
# prediction = model.predict_classes([prepare(img)])  #  removed from tensorflow 2.6
prediction = model.predict([prepare(img)])
classes = np.argmax(prediction, axis=1)

for i in classes:
    classes = categories[i]

print("expected letter: ", selected_folder)
print("predicted letter: ", classes)

X_test = np.asarray(X_test)
X_test = np.array(X_test).reshape(-1,48, 48, 1)
# y_predict = model.predict_classes(X_test)  #  removed from tensorflow 2.6
y_predict = model.predict(X_test)
y_predict = np.round(y_predict).astype(int)
y_predict = np.argmax(y_predict, axis=1)
cm = confusion_matrix(y_test, y_predict)
print(cm)
