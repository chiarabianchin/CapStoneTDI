# prediction on a new sample
import sys
import numpy as np
#time
import time
#load model
from keras.models import load_model
# keras with ImageDataGeneration
from keras.preprocessing import image
from keras.utils import plot_model

#my code
from prediction import import_labels
from train_model import populate_X_y


np.random.seed(123456789)


def prediction(model_path, dataset_path):

    start_time = time.time()

    labels = import_labels(model_path)

    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode=None)

    try:
        model = load_model(model_path)
    except:
        print "Model not found, exit"
        return
    #print test_datagen.class_indices

    plot_model(model, to_file=model_path.replace('h5', 'png'),
               show_shapes=False, show_layer_names=True, rankdir='TB')
    X_potatoes, Y_potatoes = populate_X_y('../../dataset/validation/potato/', "potato", r_w=128, r_h=128)
    X_tomatoes, Y_tomatoes = populate_X_y('../../dataset/validation/tomato/', "tomato",  r_w=128, r_h=128)
    X_train = np.concatenate((X_potatoes, X_tomatoes))
    Y_train = np.concatenate((Y_potatoes, Y_tomatoes))
    prediction = model.predict_generator(test_generator, steps=1)
    print prediction
    prediction = model.predict(X_train)
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
    print "Execution time", time.time() - start_time, "seconds"

if __name__ == "__main__":
    prediction(sys.argv[1], sys.argv[2])


'''
TODO:
- are the images in the right order?
- check the performance on the test sample done
'''