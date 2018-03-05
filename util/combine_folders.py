# combine images from many to one folder
import os
from os import listdir
from os.path import isfile, join
import sys
import shutil

def combine_folders(folders, run=False):

    for i, folder in enumerate(folders):
        files = os.listdir(folder)
        if i == 0:
            dest_folder = folder
            counter = len(files) + 1

        for f in files:
            rename = str(counter) + ".jpg"
            counter += 1

            if run:
                shutil.move(join(os.path.abspath(folder), f),
                join(os.path.abspath(dest_folder), rename))
            else:
                print join(os.path.abspath(folder), f),
                join(os.path.abspath(dest_folder), rename)
    print "Done", counter, "files in ", folders[0]

if __name__ == "__main__":
    input_folders = map(str, sys.argv[1].strip('[]').split(','))
    opt = False
    try:
        opt = bool(sys.argv[2])
    except:
        pass
    combine_folders(input_folders, run=opt)