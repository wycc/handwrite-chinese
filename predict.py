OutputFolder = 'target'
DrivePath = './'
OutputPath = DrivePath + OutputFolder

image_size=(50,50)
batch_size=100


import numpy as np
import os.path
from os import path
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
import pickle
 
 
growth_rate = 12
model=load_model('DenseNet-428.tf')
model.summary()
input_arr=[]
labels = pickle.load(open('labels','rb'))
answers=[]
for c in labels:
    answers.append(c)
succ = 0
fail = 0
for c in os.listdir('target'):

    for n in os.listdir('target/'+c):
        image = tf.keras.preprocessing.image.load_img('target/'+c+'/'+n)
        imarr = tf.keras.preprocessing.image.img_to_array(image)/255.0
        input_arr=np.array([imarr])
        ret = model.predict(input_arr)
        r = ret[0]
        idxes = np.argsort(r)
        found = False
        for i in range(1,10):
            ii = idxes[-i]
            if answers[ii] == c:
                found  = True
                break
        if found:
            succ = succ + 1
        else:
            fail = fail + 1
        print(c, fail,succ)

