import numpy as np
#model module
from keras.models import Sequential
#core layers
from keras.layers import Dense, Dropout, Activation, Flatten
#CNN layers
from keras.layers import Convolution2D, MaxPooling2D
#utilities
from keras.utils import to_categorical
#data
from keras.datasets import mnist
#plotting
from matplotlib import pyplot as plt
#save
from keras.models import load_model
from keras.models import model_from_json
# images
from PIL import Image
from keras.preprocessing import image
# file system
from os import listdir
from os.path import isfile, join
import sys
#shuffling
from sklearn.utils import shuffle

np.random.seed(123)

# Load test images
def populate_X_y(path, label):
    print("Reading from", path)
    x = []
    y = []
    for f in listdir(path):
        with open(join(path, f)) as fp:
            try:
                img = image.load_img(fp, target_size=(150, 150))
                #img = image.load_img(fp, target_size=(28, 28))
                plt.imshow(img)
                x.append(image.img_to_array(img))
                y.append(label)
                #plt.show()
            except:
                continue
    X = np.array(x)
    Y = np.array(y)

    print(X.shape, Y.shape)
    return X, Y

def convert(y_train_classes, force_classes=None):
    #covert classes to categories
    uniques, ids = np.unique(y_train_classes, return_inverse=True)
    if not force_classes:
        force_classes = len(uniques)

    Y_train = to_categorical(ids, force_classes)
    print(y_train_classes.shape, Y_train.shape)
    return Y_train

def training(X_train, y_train):

    #normalize values in the range [0,1]
    n_classes = len(set(y_train))
    print("N CLASSES", n_classes)
    X_train = X_train.astype('float32')
    X_train /= 255
    #X_test = X_test.astype('float32')
    #X_test /= 255

    print y_train.shape

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = convert(y_train)
    print("Check:", Y_train[:10])

    #model architecture
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(250,250,3)))
    #model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,3)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # if the result on the test set is much worse than the training set we should
    # increase te dropout coefficient to reduce the overfitting
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    print model.output_shape

    #compile model
    #binary_crossentropy for binary classification
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #try:
        #model.load_model("model_keras_tut.h5", by_name=True)
    #except:
        ##training
        #model.fit(X_train, Y_train,
              #batch_size=32, nb_epoch=5, verbose=1)
        #model.save("model_keras_tut.h5")
    #training
    model.fit(X_train, Y_train,
              batch_size=32, nb_epoch=5, verbose=1)
    model.save("model_keras_tut.h5")

def pred(X_test, Y_test):
    # predict

    try:
        model = load_model("model_keras_tut.h5")
        model.summary()
        model.get_config()
    except IOError:
        print "Run the training first"

    # test
    print(convert(Y_test, 2))
    score = model.evaluate(X_test, convert(Y_test, 2), verbose=0)
    print("Test score", score)
    output = model.predict(X_test)
    #print("shape output", type(output))
    for i in range(0, output.shape[0]):
        print(output[i, :], Y_test[i])

        #print np.where(output[i,:] == max(output[i,:])), Y_test[i]
def main():

    if sys.argv[1] == "train":
        X_potatoes, Y_potatoes = populate_X_y('../dataset/potatoes/', "potato")
        X_tomatoes, Y_tomatoes = populate_X_y('../dataset/tomatoes_images/', "tomato")
        X_lettuce, Y_lettuce = populate_X_y('../dataset/lettuce/', "lettuce")
        X_train = np.concatenate((X_potatoes, X_tomatoes, X_lettuce))
        Y_train = np.concatenate((Y_potatoes, Y_tomatoes, Y_lettuce))
        print(Y_train)
        print("Training...")
        training(X_train, Y_train)
        print("Done")
    else:
        X_test, Y_test = populate_X_y("../dataset/vegetable_images", '')
        pred(X_test, Y_test)

if __name__ == "__main__":
    main()