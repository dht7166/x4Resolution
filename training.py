from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,LambdaCallback
from keras.optimizers import Adam
from model import SuperResolution
from utils import *

# Some model settings
f1 = 9
f2 = 5
f3 = 5
learn_rate = 0.001
decay_plateau = 0.5

# The model
SR = SuperResolution(n1 = 64,f1 = f1,
                     n2 = 32, f2 = f2,
                     f3 = f3)
optimizer = Adam(lr = learn_rate)
SR.model.compile(loss = percep_loss,optimizer=optimizer)


# The generator
img_list = glob.glob('afhq/train/*/*.jpg')
# lvl1 generator
train_3 = Generator(img_list,32,
                    insize = 64,
                    outsize = 64*4 -f1-f2-f3+3,
                    factor = 4)

# checkpoints for various stuff

percep_ckpt = ModelCheckpoint('SR_percep_test.h5', monitor='loss',
                              save_best_only=True, mode='min')
early_stop_cb = EarlyStopping(monitor='loss',
                              min_delta=1e-6,
                              patience=3,
                              mode='min',
                              verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='loss', factor=decay_plateau, patience=0, min_lr=1e-8, verbose=1)


callback = [percep_ckpt, early_stop_cb, lr_reduce]

# Train the model on level 1
SR.model.fit(x=train_3,
             epochs=100,
             callbacks=callback)