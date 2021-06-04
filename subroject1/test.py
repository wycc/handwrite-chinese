OutputFolder = 'target'
DrivePath = './'
OutputPath = DrivePath + OutputFolder

image_size=(50,50)
batch_size=100
import pickle
def load_data():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)
    train_ds = train_datagen.flow_from_directory(
        "train",
        class_mode="categorical",
        target_size=image_size,
        batch_size=batch_size,
        subset='training'
    )
    valid_ds = train_datagen.flow_from_directory(
        "test",
        class_mode="categorical",
        target_size=image_size,
        batch_size=batch_size,
        subset='training'
    )
    print(train_ds.class_indices)
    pickle.dump(train_ds.class_indices,open('labels','wb'))
    return train_ds, valid_ds


import numpy as np
import os.path
from os import path
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.ep = 0
        print('Training start')

    def on_epoch_begin(self, batch, logs={}):
        #print(batch, logs)
        #self.losses.append(logs.get('loss'))
        optimizer = model.optimizer
        lr = K.eval(optimizer.lr)
        print('\n', ' LR: {:.6f}'.format(lr))

    def on_epoch_end(self, batch, logs={}):
        while True:
            self.ep = self.ep + 1
            if path.isdir('DenseNet-%d.tf' % self.ep):
                continue
            save_model(model, 'DenseNet-%d.tf' % self.ep,save_format='tf')
            return
        pass




import traceback 
## data
train_ds,valid_ds = load_data()
 
# model
 
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    
    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size*nb_filter, (1, 1), strides=(1,1), padding='same')(x)
    
    # Composite function
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1,1), padding='same')(x)
    
    if drop_rate: x = Dropout(drop_rate)(x)
    
    return x
 
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)
        
    return x
    
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1,1), padding='same')(x)
    if is_max != 0: x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else: x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    
    return x
 
growth_rate = 12
try:
    model=load_model('DenseNet-399.tf')
    print('Load from model')
except:
    print(traceback.format_exc())
    print('Create New Mode')
    inpt = Input(shape=(50,50,3))

    x = Conv2D(growth_rate*2, (3, 3), strides=1, padding='same')(inpt)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = BatchNormalization(axis=3)(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(4803, activation='softmax')(x)

    model = Model(inpt, x)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

model.summary()
history = LossHistory()
    
model.fit_generator(train_ds, epochs=200, validation_data=valid_ds, verbose=1,callbacks=[history])

