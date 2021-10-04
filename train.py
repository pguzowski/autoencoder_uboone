import tensorflow as tf
import numpy as np
import PIL as pil
import scipy
import skimage.measure

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Convolution2D, Activation, AveragePooling2D, Flatten, Reshape
from keras.layers import Deconvolution2D as Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import conv_block, identity_block
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras import regularizers



def pixel_weighted_loss(x_p,y):
    x=x_p[:,:,:,:1]
    weights=x_p[:,:,:,1:]
    return K.mean(weights * K.square(y - x), axis=-1)

def mse_evbyev0(x,y):
    return K.mean(K.square(y-x),axis=0)

def mse_evbyev1(x,y):
    return K.mean(K.square(y-x),axis=1)

def mse_evbyev2(x,y):
    return K.mean(K.square(y-x),axis=2)

def mse_evbyev3(x,y):
    return K.mean(K.square(y-x),axis=3)

def mse_evbyev(x,y):
    return K.mean(K.square(y-x),axis=(1,2,3))

def mse_evbyev_w(x_p,y):
    x=x_p[:,:,:,:1]
    weights=x_p[:,:,:,1:]
    return K.mean(weights * K.square(y-x),axis=(1,2,3))

base_wh = 512

input_img = Input(shape=(base_wh, base_wh, 1))  # adapt this if using `channels_first` image data format

if True:
    x = ZeroPadding2D((3, 3))(input_img)
    print x.name, x.get_shape()
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    print x.name, x.get_shape()
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    print x.name, x.get_shape()
    x = Activation('relu')(x)
    print x.name, x.get_shape()
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    print x.name, x.get_shape()

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    print x.name, x.get_shape()
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    print x.name, x.get_shape()

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    print x.name, x.get_shape()

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    print x.name, x.get_shape()

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    print x.name, x.get_shape()

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    print x.name, x.get_shape()

    x = Flatten()(x)
    print x.name, x.get_shape()
    x = Dense(2*32*32)(x)
    print x.get_shape()

    encoded = x


    #decoded = Reshape((32,32,2))(x)

    x = Dense(2*2*2048)(x)
    print x.name, x.get_shape()
    x = Reshape((2,2,2048))(x)
    print x.name, x.get_shape()

    x = Conv2DTranspose(2048,1,1,(None,16,16,2048),subsample=(8,8))(x)
    print x.name, x.get_shape()
    
    x = conv_block(x, 3, [512, 512, 2048], strides=(1,1), stage=6, block='a')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [512, 512, 2048], stage=6, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [512, 512, 2048], stage=6, block='c')
    print x.name, x.get_shape()

    x = Conv2DTranspose(1024,1,1,(None,32,32,1024),subsample=(2,2))(x)
    print x.name, x.get_shape()

    x = conv_block(x, 3, [256, 256, 1024], strides=(1,1), stage=7, block='a')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='c')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='d')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='e')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [256, 256, 1024], stage=7, block='f')
    print x.name, x.get_shape()

    x = Conv2DTranspose(512,1,1,(None,64,64,512),subsample=(2,2))(x)
    print x.name, x.get_shape()

    x = conv_block(x, 3, [128, 128, 512], stage=8, strides=(1,1), block='a')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [128, 128, 512], stage=8, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [128, 128, 512], stage=8, block='c')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [128, 128, 512], stage=8, block='d')
    print x.name, x.get_shape()

    x = Conv2DTranspose(256,1,1,(None,128,128,256),subsample=(2,2))(x)
    print x.name, x.get_shape()

    x = conv_block(x, 3, [64, 64, 256], stage=9, block='a', strides=(1, 1))
    print x.name, x.get_shape()
    x = identity_block(x, 3, [64, 64, 256], stage=9, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [64, 64, 256], stage=9, block='c')
    print x.name, x.get_shape()

    x = Conv2DTranspose(128,1,1,(None,256,256,128),subsample=(2,2))(x)
    print x.name, x.get_shape()

    x = conv_block(x, 3, [32, 32, 128], stage=10, block='a', strides=(1, 1))
    print x.name, x.get_shape()
    x = identity_block(x, 3, [32, 32, 128], stage=10, block='b')
    print x.name, x.get_shape()
    x = identity_block(x, 3, [32, 32, 128], stage=10, block='c')
    print x.name, x.get_shape()

    x = Conv2DTranspose(64,1,1,(None,512,512,64),subsample=(2,2))(x)
    print x.name, x.get_shape()

    x = ZeroPadding2D((3, 3))(x)
    print x.name, x.get_shape()
    x = Convolution2D(64, 7, 7, subsample=(1, 1), name='conv3')(x)
    print x.name, x.get_shape()
    x = ZeroPadding2D((3, 3))(x)
    print x.name, x.get_shape()
    x = Convolution2D(3, 7, 7, subsample=(1, 1), name='conv4')(x)
    print x.name, x.get_shape()
    x = ZeroPadding2D((3, 3))(x)
    print x.name, x.get_shape()
    x = Convolution2D(1, 7, 7, subsample=(1, 1), name='conv5')(x)
    print x.name, x.get_shape()

    #x = Activation('softmax')(x)
    #print x.name, x.get_shape()

    decoded = x


autoencoder = Model(input_img, decoded,)
autoencoder.compile(
        #optimizer='adadelta',
        optimizer=RMSprop(lr=0.0003),
        #optimizer=SGD(lr=0.1, decay=1e-6, momentum=1.9),
        #loss='mse',
        #loss='binary_crossentropy',
        loss=pixel_weighted_loss,
        #metrics=[mse_evbyev,mse_evbyev1,mse_evbyev2,mse_evbyev3,mse_evbyev4]
        metrics=[mse_evbyev_w]
        )




def _parse_function(filename):
    X=np.load(filename)['plane2'].reshape((1,))[0]
    
    z00 = X.astype(np.float32).toarray().reshape((3456,1008,1));
    while True:
        i = np.random.randint(3456-base_wh)
        j = np.random.randint(1008-base_wh)
        z0 = z00[i:i+base_wh,j:j+base_wh,:]
        if z0.max() > 0. or z0.min() < 0.: break
    #print 'z0 shape:', z0.shape
    z = z0
    if z.max() > z.min():    z = (z0-np.min(z0))/(np.max(z0)-np.min(z0))
    #zwh,edg = np.histogram(z0,bins=[0,1,13])
    maxneg=-0.5
    minpos=0.5
    #print z0.min(),z0.max(),z0[z0<0.].shape,z0[z0>0.].shape
    if z0.min()<0.: maxneg = np.max(z0[z0<0.])
    if z0.max()>0.: minpos = np.min(z0[z0>0.])
    zwh,edg = np.histogram(z0,bins=[-5000,maxneg/2,minpos/2,5000])
    zwh=zwh.sum().astype(np.float32)/(zwh+1e-10)
    zw = np.piecewise(z0,[(z0>=edg[i]-0.5)&(z0<edg[i+1]-0.5) for i in xrange(len(edg)-1)],zwh)
    sumw = np.sum(zw) / zw.shape[0] / zw.shape[1]
    return z, np.dstack([z,zw/sumw])


def randint(filename):
    X=np.load(filename)['plane2'].reshape((1,))[0]
    
    z00 = X.astype(np.float32).toarray().reshape((3456,1008,1));
    i = np.random.randint(3456-base_wh)
    j = np.random.randint(1008-base_wh)
    while True:
        z0 = z00[i:i+base_wh,j:j+base_wh,:]
        if z0.max() > 0. or z0.min() < 0.: break
        i = np.random.randint(3456-base_wh)
        j = np.random.randint(1008-base_wh)
    return (i, j)

def _parse_function_v(arg):
    filename,(i,j) = arg
    X=np.load(filename)['plane2'].reshape((1,))[0]
    
    z0 = X.astype(np.float32).toarray().reshape((3456,1008,1));
    z0 = z0[i:i+base_wh,j:j+base_wh,:]
    z = z0
    if z.max() > z.min():    z = (z0-np.min(z0))/(np.max(z0)-np.min(z0))
    #zwh,edg = np.histogram(z0,bins=[0,1,13])
    maxneg=-0.5
    minpos=0.5
    if z0.min()<0.: maxneg = np.max(z0[z0<0.])
    if z0.max()>0.: minpos = np.min(z0[z0>0.])
    zwh,edg = np.histogram(z0,bins=[-5000,maxneg/2,minpos/2,5000])
    zwh=zwh.sum().astype(np.float32)/(zwh+1e-10)
    zw = np.piecewise(z0,[(z0>=edg[i]-0.5)&(z0<edg[i+1]-0.5) for i in xrange(len(edg)-1)],zwh)
    sumw = np.sum(zw) / zw.shape[0] / zw.shape[1]
    return z, np.dstack([z,zw/sumw])

if False:
    #z = (z0+4096.)/4096./2.
    z = (z0-np.min(z0))/(np.max(z0)-np.min(z0))
    zz = skimage.measure.block_reduce(z,(6,2),np.max)
    zz2 = skimage.measure.block_reduce(z,(6,2),np.min)
    zzm = skimage.measure.block_reduce(z,(6,2),np.mean)
    zzw = skimage.measure.block_reduce(z0,(6,2),np.count_nonzero)
    zzwh,edg = np.histogram(zzw,bins=[0,1,5,13])
    zzwh = zzwh.sum().astype(np.float32)/(zzwh+1e-10)
    #zzwh[0] = zzwh[0]/100.
    zzw = zzw.astype(np.float32)
    zzw = np.piecewise(zzw,[(zzw>=edg[i]-0.5)&(zzw<edg[i+1]-0.5) for i in xrange(len(edg)-1)],zzwh)
    #zzw = v_reweight(x=zzw,hist=zzwh,bins=edg)
    sumw = np.sum(zzw) / zzw.shape[0] / zzw.shape[1] 
    zzw = zzw / sumw
    
    zz3 = np.dstack([zz,zz2,zzm])
    zz4 = np.dstack([zz,zz2,zzm,zzw])

    #return zz3,zz4

# A vector of filenames.
import os
filenames = ['output7/%s'%f for f in os.listdir('output7') if f.endswith('.npz') ]
valid_filenames = ['outputV/%s'%f for f in os.listdir('outputV') if f.endswith('.npz') ]
valid_starts = [randint(f) for f in valid_filenames]
np.random.shuffle(filenames)

epochs=350
steps_per_epoch=25
batch_size=4
valid_batch_size=4
valid_steps=640/valid_batch_size
min_mean_valid_loss = 1e10000
alllosses=[]
try:
    for epoch in xrange(epochs):
        for step in xrange(steps_per_epoch):
            startev = (epoch * steps_per_epoch + step * batch_size) % len(filenames)
            stopev = (epoch * steps_per_epoch + (step+1) * batch_size) % len(filenames)
            if(startev > stopev):
                a = filenames[startev:]
                np.random.shuffle(filenames)
                dataset=map(_parse_function,filenames[:stopev]+a)
            else:
                dataset=map(_parse_function,filenames[startev:stopev])
            x,y = zip(*dataset)
            loss = autoencoder.train_on_batch(np.stack(x),np.stack(y))
            #print loss
            #print loss[1].shape
            #print loss[2].shape
            #print loss[3].shape
            #print loss[4].shape
            #print loss[5].shape
            #print len(y)
            #print len(dataset)
            #print np.stack(y).shape
            #raise Exception
            #print epoch, step, loss
        mean_valid_loss = 0.;
        alllosses=[]
        for step in xrange(valid_steps):
            startev = (step * valid_batch_size) % len(valid_filenames)
            stopev = ((step+1) * valid_batch_size) % len(valid_filenames)
            if(startev > stopev):
                dataset=map(_parse_function_v,zip(valid_filenames[:stopev]+valid_filenames[startev:],valid_starts[:stopev]+valid_starts[startev:]))
            else:
                dataset=map(_parse_function_v,zip(valid_filenames[startev:stopev],valid_starts[startev:stopev]))
            x,y = zip(*dataset)
            losses=autoencoder.test_on_batch(np.stack(x),np.stack(y))
            mean_valid_loss+=losses[0]
            alllosses+=[losses[1]]
        print epoch,'VALID',mean_valid_loss/valid_steps#,alllosses
        if mean_valid_loss < min_mean_valid_loss:
            min_mean_valid_loss = mean_valid_loss
            autoencoder.save('autoencoder.min.mdl')
            np.save('alllosses.min.npy',np.concatenate(alllosses))
except KeyboardInterrupt:
    pass
finally:
    autoencoder.save('autoencoder.mdl')
    if len(alllosses) >0:    np.save('alllosses.npy',np.concatenate(alllosses))

#print dataset
#print dataset
#autoencoder.fit(x,y,epochs=50,steps_per_epoch=25,validation_data = (xv,yv),validation_steps=10)


if False:
    input_img = Input(shape=(576, 504, 3))  # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    #encoded = MaxPooling2D((2, 2), padding='same')(x)

    #print encoded.shape

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    decoded = Cropping2D(cropping=(0,4),data_format='channels_last')(x)

    autoencoder = Model(input_img, decoded,)
    autoencoder.compile(
            optimizer='adadelta',
            #optimizer=SGD(lr=0.1, decay=1e-6, momentum=1.9),
            #loss='mse',
            #loss='binary_crossentropy',
            loss=pixel_weighted_loss,
            )

# 2018 09 18
if False:
        
    print input_img.get_shape()
    x = Conv2D(8, 7, 7, activation='relu',border_mode='same')(input_img)
    print x.get_shape()
    x = MaxPooling2D((2, 2))(x)
    print x.get_shape()
    x = Conv2D(16, 7, 7, activation='relu',border_mode='same')(x)
    print x.get_shape()
    x = MaxPooling2D((2, 2))(x)
    print x.get_shape()
    x = Conv2D(24, 7, 7, activation='relu',border_mode='same')(x)
    print x.get_shape()
    x = MaxPooling2D((2, 2))(x)
    print x.get_shape()
    x = Conv2D(32, 7, 7, activation='relu',border_mode='same')(x)
    print x.get_shape()
    encoded = MaxPooling2D((2, 2))(x)
    print encoded.get_shape()

    x = Conv2D(32, 7, 7, activation='relu',border_mode='same')(encoded)
    print x.get_shape()
    x = UpSampling2D((2, 2))(x)
    print x.get_shape()
    x = Conv2D(24, 7, 7, activation='relu',border_mode='same')(x)
    print x.get_shape()
    x = UpSampling2D((2, 2))(x)
    print x.get_shape()
    x = Conv2D(16, 7, 7, activation='relu',border_mode='same')(x)
    print x.get_shape()
    x = UpSampling2D((2, 2))(x)
    print x.get_shape()
    x = Conv2D(8, 7, 7, activation='relu',border_mode='same')(x)
    print x.get_shape()
    x = UpSampling2D((2, 2))(x)
    print x.get_shape()
    decoded = Conv2D(1, 7, 7, activation='sigmoid',border_mode='same')(x)
    print decoded.get_shape()


