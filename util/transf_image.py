# transform image in matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

f = '../dataset/train/tomatoes_images/47.jpg'
img = None
try:
    img = Image.open(f)
except:
    os.remove(f)

array = np.array(img)
print(array.shape)

plt.imshow(array)
plt.show()