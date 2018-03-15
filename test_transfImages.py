import sys
from train_model import populate_X_y

def main(path, labels):
    populate_X_y(path, labels)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])