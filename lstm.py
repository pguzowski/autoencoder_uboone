import numpy as np
import tensorflow as tf

p = np.load('flat/32bit_flat.npz')['arr_0']

ngen=200000
ntrain = int(ngen*0.66)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(1024, activation='relu', input_shape=(63,1)))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy','mae'])

test_X = None;
test_y = None;

for loopi in range(1):

    pp0=np.sort(np.random.choice(p,(ngen,2)).astype('uint64'))
    p0 = pp0[:,0]

    pp=p0*pp0[:,1]

    bb = [pp%(2**i)//(2**(i-1)) for i in range(2,65)]
    b0 = [p0%(2**i)//(2**(i-1)) for i in range(2,33)]

    bits=np.concatenate([bb,b0]).astype('int8').transpose()

    X = bits[:,:63]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = bits[:,63]

    train_X = X[:ntrain]
    test_X = X[ntrain:] if test_X is None else np.concatenate([test_X,X[ntrain:]])
    train_y = y[:ntrain]
    test_y = y[ntrain:] if test_y is None else np.concatenate([test_y,y[ntrain:]])


    model.fit(train_X, train_y, epochs=1500, batch_size=4000, validation_split=0.3, callbacks=[es], verbose=1)
    loss, acc = model.evaluate(test_X, test_y, verbose=0, batch_size=4000)
    print('loop %d Test Accuracy: %.3f' % (loopi,acc))
