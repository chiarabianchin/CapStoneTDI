# clean up the image folder removing the files containing corrupted images
import os
from os import listdir
from os.path import isfile, join
import sys
from keras.preprocessing import image

def cleanup(path, run=False):
    print("Reading from", path)

    for root, dirs, ff in os.walk(path):
        for d in dirs:
            folder_path = join(path, d)
            print "Examining", folder_path
            count = 0
            for _, _, f in os.walk(folder_path):
                if len(f) == 0:
                    break
                print "N Files, no dir", len(f)
                for i in f:
                    with open(join(folder_path, i)) as fp:
                        try:
                            image.load_img(fp, target_size=(250, 250))

                        except:
                            count+=1

                            print "Remove", fp
                            if run:
                                os.remove(join(folder_path, i))
            print "Files removed: ", count
            print "--------------------------------------------"

if __name__ == "__main__":
    opt = False
    try:
        opt = sys.argv[2]
    except:
        pass
    cleanup(sys.argv[1], run=opt)