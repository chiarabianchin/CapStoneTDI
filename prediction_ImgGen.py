# prediction on a new sample
import numpy as np
#time
import time
#load model
from keras.models import load_model
# keras with ImageDataGeneration
from keras.preprocessing import image
from keras.utils import plot_model

np.random.seed(123456789)


def prediction():

    start_time = time.time()

    labels = {0: 'butter', 2: 'lettuce', 1: 'eggs', 3: 'tomatoes_images',
              4: 'zucchini'}

    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        '../dataset/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None)

    try:
        model = load_model("model_32_steps_epoch_20_epochs_10_val_steps.h5")
    except:
        print "Model not found, exit"
        return
    #print test_datagen.class_indices

    plot_model(model, to_file='model.png', show_shapes=False,
               show_layer_names=True, rankdir='TB')
    prediction = model.predict_generator(test_generator, steps=1)
    print prediction
    y_classes = prediction.argmax(axis=-1)
    y = 0
    n = 0
    for i, im in enumerate(test_generator.filenames):
        print im, '\t\t\t', labels[y_classes[i]]
        if im.find(labels[y_classes[i]]) == -1:
            n += 1
        else:
            y += 1
    print "Correct answers", y, "; Wrong answers: ", n
    print "Expected Accuracy",
    print "Execution time", time.time()-start_time, "seconds"

if __name__ == "__main__":
    prediction()


'''
TODO:
- are the images in the right order?
- check the performance on the test sample done
'''