#prediction
#general
import sys
import numpy as np
import itertools
#time
import time
#load model
from keras.models import load_model
# keras with ImageDataGeneration
from keras.preprocessing import image
from keras.utils import plot_model, to_categorical
#plotting
from matplotlib import pyplot as plt
#save
from keras.models import load_model
from keras.models import model_from_json
# images
from PIL import Image
#sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,\
                            precision_score, recall_score

#my methods
from train_model import populate_X_y

def convert(y_train_classes, force_classes=None):
    #covert classes to categories
    #TODO fix the determination of number of classes: from dictionary of labels, to be saved
    uniques, ids = np.unique(y_train_classes, return_inverse=True)
    if not force_classes:
        force_classes = len(uniques)
    print y_train_classes, uniques, ids
    Y_train = to_categorical(ids, force_classes)
    print(y_train_classes.shape, Y_train.shape)
    return Y_train, ids

def pred(X_test, Y_test, model_path, n_cl):
    # predict

    try:
        model = load_model(model_path)
        model.summary()
        model.get_config()
    except IOError:
        print "Run the training first"

    # test
    #print(convert(Y_test, 2))
    print "READ HERE"
    Y_test_cat, Y_test_cl = convert(Y_test, n_cl)
    score = model.evaluate(X_test, Y_test_cat, verbose=0)
    #score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test score", score)
    output = model.predict(X_test)
    #print("shape output", type(output))
    for i in range(0, output.shape[0]):
        print output[i, :], Y_test[i]

        Y_pred = output.argmax(axis=-1)
        print np.where(output[i,:] == max(output[i,:]))

    print "true ", Y_test_cat.shape, "pred", len(Y_pred)

    cm = confusion_matrix(Y_test_cl, Y_pred)
    print "Accuracy", accuracy_score(Y_test_cl, Y_pred)
    print "Confusion matrix", cm
    print "Precision", precision_score(Y_test_cl, Y_pred)
    print "Recall", recall_score(Y_test_cl, Y_pred)
    print "F1", f1_score(Y_test_cl, Y_pred)

    plt.figure()
    plot_confusion_matrix(cm, set(Y_test))
    plt.show()#save("confusion_matrix.pdf")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def main(model_path, test_paths, labels, n_cl):
    X_test = []
    Y_test = []

    for i, p in enumerate(test_paths):
        X, Y = populate_X_y(p, labels[i])
        if len(X_test) > 0:
            X_test = np.concatenate((X_test, X))
            Y_test = np.concatenate((Y_test, Y))
        else:
            X_test = X
            Y_test = Y
    print X_test.shape, Y_test.shape

    pred(X_test, Y_test, model_path, n_cl)

if __name__ == "__main__":
    model_path = sys.argv[1]
    test_paths = map(str, sys.argv[2].strip('[]').split(','))
    labels =  map(str, sys.argv[3].strip('[]').split(','))
    n_cl = int(sys.argv[4])
    main(model_path, test_paths, labels, n_cl)