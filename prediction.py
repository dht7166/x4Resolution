from model import SuperResolution
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Running x4 Super Resolution on Image")
parser.add_argument('weight',help = 'path to weight file')
parser.add_argument('input',help = 'path to input image')
parser = parser.parse_args()



f1 = 9
f2 = 5
f3 = 5


# The model
SR = SuperResolution(n1 = 64,f1 = f1,
                     n2 = 32, f2 = f2,
                     f3 = f3)
SR.model.summary()
SR.model.load_weights(parser.weight)
img = cv2.imread(parser.input)
H,W,C = img.shape
img = cv2.resize(img,(H*4,W*4))
img = img/255
img = img[np.newaxis,:,:,:]
prediction = SR.model.predict(img)
prediction = prediction[0,:,:,:]
prediction = (prediction*255).astype(np.uint8)
cv2.imwrite(parser.input[:-4]+'_x4.jpg',prediction)

