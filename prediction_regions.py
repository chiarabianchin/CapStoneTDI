# general
import sys

# my code
from prediction import run_model_on_proposed_regions


def main(*args):
    # path_img, model, r_w=150, r_h=150, n=100, opt='f'
    run_model_on_proposed_regions(*args)

if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    rw, rh = 150, 150
    if len(sys.argv) > 2:
        rw, rh = int(sys.argv[3]), int(sys.argv[4])
    main(image_path, model_path, rw, rh)