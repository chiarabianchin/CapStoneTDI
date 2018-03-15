import numpy as np
import sys
import json
#time
import time
#model module
from keras.models import Sequential
#core layers
from keras.layers import Dense, Dropout, Activation, Flatten
#CNN layers
from keras.layers import Convolution2D, MaxPooling2D
#utilities
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import History, TensorBoard, EarlyStopping
from keras import optimizers
# keras with ImageDataGeneration
from keras.preprocessing import image
#save
from keras.models import load_model
#plotting
from matplotlib import pyplot as plt

def train_model(train_path, val_path, model_f_name, isbinary=False):
    '''
    train_path = ../../dataset/train/
    val_path = ../../dataset/validation/
    model_f_name = model_tr6400_5_v1343_5_steps_per_epoch_32_epochs_20_validation_steps_10
    isbinary=False
    '''
    start_time = time.time()
    np.random.seed(123456789)

    # tensor flow board.Open with > tensorboard --logdir="logs"
    board = TensorBoard(log_dir='./logs/{}'.format(time.time()),
                histogram_freq=0, batch_size=32, write_graph=True,
                write_grads=False, write_images=False, embeddings_freq=0,
                embeddings_layer_names=None, embeddings_metadata=None)
    stop_when_ok = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3,
                                 verbose=1, mode='auto')
    callbacks_list = [stop_when_ok, board]
    # read images for training sample
    # data augmentation applied
    im_w = 128  # 150
    im_h = 128  # 150
    im_c = 3
    train_datagen = image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(im_w, im_h),
            batch_size=32,
            class_mode='categorical')

    json.dump(train_generator.class_indices, open(model_f_name + '.json', 'w'))

    # read images for validation sample
    validation_generator = test_datagen.flow_from_directory(
            val_path,
            target_size=(im_w, im_h),
            batch_size=32,
            class_mode='categorical')

    #define model
    model = Sequential()
    model.add(Convolution2D(32, 7, 7, activation='relu',
                            input_shape=(im_w, im_h, im_c)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # this additional convolution doesn't seem to help
    model.add(Convolution2D(64, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # if the result on the test set is much worse than the training set we should
    # increase te dropout coefficient to reduce the overfitting
    #model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #TODO: test with reduced Dropout, e.g. 0.25 to check if the training accuracy gets
    #higher than the validation accuracy
    model.add(Dropout(0.5))
    model.add(Dense(len(train_generator.class_indices), activation='softmax'))
    adam = optimizers.Adam(lr=1e-3)  # default 1e-3
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    output = None
    #print model.to_json()

    # read the model from file or fit it
    try:
        model = load_model("dummy")

    except:
        output = model.fit_generator(
            train_generator,
            steps_per_epoch=32,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=10,
            callbacks=callbacks_list)
    #save figure of the model structure
    plot_model(model, to_file=model_f_name+'.png', show_shapes=False,
               show_layer_names=True, rankdir='TB')
    print "________________________________________"
    print model.predict_generator(validation_generator)
    # output of the performance at each iteration (epoch)
    performance = output.history
    print(performance)
    #save the model to file
    model.save(model_f_name + ".h5")

    print "Execution time", time.time() - start_time, "seconds"
    # plot performance
    plt.figure(1)
    plt.subplot(121)
    plt.plot(range(1, len(performance['acc']) + 1), performance['acc'], 'r*')
    plt.plot(range(1, len(performance['val_acc']) + 1), performance['val_acc'],
             'bo')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(['train', 'validation'])

    plt.subplot(122)
    plt.plot(range(1, len(performance['loss']) + 1), performance['loss'], 'r*')
    plt.plot(range(1, len(performance['val_loss']) + 1), performance['val_loss'],
              'bo')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'])
    plt.savefig("performance_" + model_f_name + ".pdf")

    #TODO:
    '''
    - prediction with current model, done
    - train with more images
    - test another model adding a convolution for instance
    - implement patience in training,
    - calculate precision TP/(FP + TP) and recall TP/(TP+FN), confusion matrix
    - recongize complex image with more vegetables
    - add images
      '''

    plt.show()


def main():
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    model_f_name = sys.argv[3]
    isbinary = False
    try:
        isbinary = bool(sys.argv[4])
    except:
        pass
    train_model(train_path, val_path, model_f_name, isbinary=isbinary)

if __name__ == "__main__":
    main()
