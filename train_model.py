import numpy as np
#model module
from keras.models import Sequential
#core layers
from keras.layers import Dense, Dropout, Activation, Flatten
#CNN layers
from keras.layers import Convolution2D, MaxPooling2D
#utilities
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import History, TensorBoard
# keras with ImageDataGeneration
from keras.preprocessing import image
#save
from keras.models import load_model
#plotting
from matplotlib import pyplot as plt

np.random.seed(123)

history = History()

# tensor flow board.Open with > tensorboard --logdir="logs"
board = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,\
            write_grads=False, write_images=False, embeddings_freq=0, \
            embeddings_layer_names=None, embeddings_metadata=None)

# read images for training sample
# data augmentation applied
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '../dataset/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
print(train_generator.class_indices)

# read images for validation sample
validation_generator = test_datagen.flow_from_directory(
        '../dataset/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

#define model
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(150,150,3)))
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,3)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# if the result on the test set is much worse than the training set we should
# increase te dropout coefficient to reduce the overfitting
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
output = None
# read the model from file or fit it
try:
    model = load_model("model_20_steps_epoch_20_epochs_20_val_steps.h5", by_name=True)

except:
    output = model.fit_generator(
        train_generator,
        steps_per_epoch=32,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[board])
#save figure of the model structure
plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True,
           rankdir='TB')

# output of the performance at each iteration (epoch)
performance = output.history
print(performance)
#save the model to file
model.save("model_testing.h5")

# plot performance
plt.figure(1)
plt.subplot(121)
plt.plot(range(1, len(performance['acc'])+1), performance['acc'], 'r*')
plt.plot(range(1, len(performance['val_acc'])+1), performance['val_acc'], 'bo')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend(['train', 'validation'])

plt.subplot(122)
plt.plot(range(1, len(performance['loss'])+1), performance['loss'], 'r*')
plt.plot(range(1, len(performance['val_loss'])+1), performance['val_loss'], 'bo')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'validation'])


#TODO:
'''
- prediction with current model, done
- train with more images
- test another model adding a convolution for instance
- implement patience in training,
- recongize complex image with more vegetables
- add images
  '''

plt.show()


