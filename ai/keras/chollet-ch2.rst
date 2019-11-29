

::

    [blyth@localhost ~]$ ip
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: from keras.datasets import mnist
    /home/blyth/anaconda2/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.

    In [2]: (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    In [3]: train_images = train_images.reshape((60000, 28 * 28))

    In [4]: from keras import models

    In [5]: from keras import layers

    In [6]: network = models.Sequential()

    In [7]: network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

    In [8]: network.add(layers.Dense(10, activation='softmax'))
       ...: 

    In [9]: network.compile(optimizer='rmsprop',
       ...:                 loss='categorical_crossentropy',
       ...:                 metrics=['accuracy'])
       ...: 

    In [10]: train_images = train_images.astype('float32') / 255

    In [11]: test_images = test_images.reshape((10000, 28 * 28))

    In [12]: test_images = test_images.astype('float32') / 255
        ...: 

    In [13]: from keras.utils import to_categorical


    In [15]: to_categorical?
    Signature: to_categorical(y, num_classes=None)
    Docstring:
    Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    File:      ~/anaconda2/lib/python2.7/site-packages/keras/utils/np_utils.py
    Type:      function

    In [16]: train_labels = to_categorical(train_labels)

    In [17]: test_labels = to_categorical(test_labels)

    In [18]: train_labels
    Out[18]: 
    array([[0., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 1., 0.]], dtype=float32)

    In [19]: train_labels.shape
    Out[19]: (60000, 10)

    In [20]: test_labels
    Out[20]: 
    array([[0., 0., 0., ..., 1., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

    In [21]: test_labels.shape
    Out[21]: (10000, 10)

    In [22]: network.fit(train_images, train_labels, epochs=5, batch_size=128)
    Epoch 1/5
    2019-07-05 20:54:10.182499: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
    2019-07-05 20:54:10.195689: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
    60000/60000 [==============================] - 5s 89us/step - loss: 0.2554 - acc: 0.9267
    Epoch 2/5
    60000/60000 [==============================] - 5s 82us/step - loss: 0.1038 - acc: 0.9689
    Epoch 3/5
    60000/60000 [==============================] - 5s 79us/step - loss: 0.0677 - acc: 0.9797
    Epoch 4/5
    60000/60000 [==============================] - 5s 79us/step - loss: 0.0505 - acc: 0.9851
    Epoch 5/5
    60000/60000 [==============================] - 5s 79us/step - loss: 0.0378 - acc: 0.9890
    Out[22]: <keras.callbacks.History at 0x7f9db93bf790>

    In [23]: test_loss, test_acc = network.evaluate(test_images, test_labels)
    10000/10000 [==============================] - 1s 81us/step

    In [24]: test_loss
    Out[24]: 0.06604315490316366

    In [25]: test_acc
    Out[25]: 0.9804

    In [26]: test_images
    Out[26]: 
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)


    In [35]: network.count_params?
    Signature: network.count_params()
    Docstring:
    Counts the total number of scalars composing the weights.

    # Returns
        An integer count.

    # Raises
        RuntimeError: if the layer isn't yet built
            (in which case its weights aren't yet defined).
    File:      ~/anaconda2/lib/python2.7/site-packages/keras/engine/topology.py
    Type:      instancemethod

    In [36]: network.count_params()
    Out[36]: 407050

    In [44]: network.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    _________________________________________________________________

    In [45]: network.predict?
    Signature: network.predict(x, batch_size=None, verbose=0, steps=None)
    Docstring:
    Generates output predictions for the input samples.

    The input samples are processed batch by batch.

    # Arguments
        x: the input data, as a Numpy array.
        batch_size: Integer. If unspecified, it will default to 32.
        verbose: verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`.

    # Returns
        A Numpy array of predictions.
    File:      ~/anaconda2/lib/python2.7/site-packages/keras/models.py
    Type:      instancemethod

    In [46]: network.predict(test_images)
    Out[46]: 
    array([[5.0690559e-11, 8.8978573e-12, 2.7054725e-07, ..., 9.9999893e-01,
            6.2740824e-10, 4.0344247e-08],
           [1.9880503e-11, 4.0226269e-07, 9.9999952e-01, ..., 3.0721254e-18,
            5.3364456e-08, 7.2550098e-16],
           [8.1046146e-08, 9.9829429e-01, 1.3006509e-04, ..., 1.2582332e-03,
            2.2433448e-04, 6.0722941e-06],
           ...,
           [1.6602235e-12, 3.5617560e-09, 2.8954150e-10, ..., 2.5061102e-04,
            3.6996746e-06, 3.9551701e-04],
           [7.6731009e-12, 7.3127407e-11, 4.5778611e-12, ..., 4.3104639e-12,
            8.0173377e-06, 6.6441015e-12],
           [4.0082833e-11, 1.0652506e-14, 2.1424655e-11, ..., 4.5058061e-16,
            4.0855214e-12, 4.2057771e-14]], dtype=float32)

    In [47]: p = network.predict(test_images)

    In [48]: p.shape
    Out[48]: (10000, 10)

    In [49]: p[0]
    Out[49]: 
    array([5.0690559e-11, 8.8978573e-12, 2.7054725e-07, 7.4583750e-07,
           2.0483063e-15, 1.1129092e-09, 6.3399855e-16, 9.9999893e-01,
           6.2740824e-10, 4.0344247e-08], dtype=float32)

    In [50]: p[0].argmax()
    Out[50]: 7

    In [51]: p[1].argmax()
    Out[51]: 2

    In [52]: p[2].argmax()
    Out[52]: 1

    In [53]: test_labels
    Out[53]: 
    array([[0., 0., 0., ..., 1., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

    In [54]: test_labels[0].argmax()
    Out[54]: 7

    In [55]: test_labels[1].argmax()
    Out[55]: 2

    In [56]: test_labels[2].argmax()
    Out[56]: 1

    In [57]: test_labels.argmax(axis=1)
    Out[57]: array([7, 2, 1, ..., 4, 5, 6])

    In [58]: test_labels.argmax(axis=1).shape
    Out[58]: (10000,)

    In [59]: pp = p.argmax(axis=1)

    In [60]: pp
    Out[60]: array([7, 2, 1, ..., 4, 5, 6])

    In [61]: tt = test_labels.argmax(axis=1)

    In [62]: np.all( pp == tt )
    Out[62]: False

    In [63]: np.where( pp != tt )
    Out[63]: 
    (array([  18,  217,  247,  321,  340,  381,  445,  447,  495,  582,  610,
             619,  659,  684,  691,  717,  720,  846,  947,  951,  965, 1014,
            1039, 1112, 1178, 1181, 1182, 1226, 1232, 1242, 1247, 1260, 1299,
            1319, 1393, 1500, 1522, 1530, 1549, 1553, 1581, 1609, 1626, 1681,
            1754, 1790, 1878, 1901, 1982, 1984, 2004, 2016, 2053, 2098, 2109,
            2118, 2130, 2135, 2182, 2224, 2293, 2329, 2387, 2414, 2454, 2488,
            2607, 2618, 2635, 2648, 2654, 2743, 2810, 2860, 2863, 2877, 2915,
            2921, 2927, 2939, 3060, 3073, 3117, 3289, 3330, 3405, 3422, 3503,
            3520, 3533, 3542, 3558, 3565, 3567, 3718, 3751, 3776, 3780, 3796,
            3808, 3818, 3838, 3853, 3902, 3906, 3941, 3968, 3985, 4065, 4078,
            4176, 4199, 4224, 4248, 4271, 4289, 4294, 4306, 4360, 4497, 4534,
            4536, 4571, 4578, 4639, 4731, 4807, 4823, 4833, 4860, 4880, 4990,
            5331, 5457, 5600, 5642, 5676, 5734, 5842, 5887, 5936, 5937, 5955,
            5972, 5973, 5981, 5982, 5997, 6009, 6011, 6023, 6045, 6046, 6059,
            6166, 6555, 6571, 6574, 6576, 6597, 6625, 6651, 6755, 6783, 7121,
            7216, 7800, 7821, 7921, 8094, 8277, 8311, 8325, 8408, 8527, 9009,
            9015, 9024, 9422, 9538, 9587, 9634, 9664, 9669, 9679, 9692, 9698,
            9729, 9745, 9768, 9770, 9782, 9792, 9811, 9839, 9858]),)

    In [64]: 

