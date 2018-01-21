"""
Convolutional Neural Network test on cats and dogs data.

WARNING: The cats and dogs dataset is TOO LARGE to push to Github. Download at
https://www.superdatascience.com/machine-learning/ , Part 8. Deep Learning,
Convolutional Neural Networks.zip Note the directory structure for this test:
python.ml
    dataset
        test
        training
WARNING: For Python 3.6 install Tensorflow 1.5+ Note the version number in the url!
INSTALL: pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0rc1-py3-none-any.whl

The test takes about 8+ minutes:
Using TensorFlow backend.
Found 8000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
Epoch 1/10
2018-01-21 14:44:57.636634: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX
100/100 [==============================] - 44s 438ms/step - loss: 0.6997 - acc: 0.5416 - val_loss: 0.6597 - val_acc: 0.6030
Epoch 2/10
100/100 [==============================] - 46s 464ms/step - loss: 0.6615 - acc: 0.6059 - val_loss: 0.6314 - val_acc: 0.6790
Epoch 3/10
100/100 [==============================] - 44s 439ms/step - loss: 0.6266 - acc: 0.6497 - val_loss: 0.6587 - val_acc: 0.6035
Epoch 4/10
100/100 [==============================] - 45s 449ms/step - loss: 0.5987 - acc: 0.6784 - val_loss: 0.5696 - val_acc: 0.7100
Epoch 5/10
100/100 [==============================] - 49s 489ms/step - loss: 0.5884 - acc: 0.6869 - val_loss: 0.6741 - val_acc: 0.6320
Epoch 6/10
100/100 [==============================] - 46s 457ms/step - loss: 0.5772 - acc: 0.7000 - val_loss: 0.5479 - val_acc: 0.7360
Epoch 7/10
100/100 [==============================] - 48s 479ms/step - loss: 0.5578 - acc: 0.7119 - val_loss: 0.5346 - val_acc: 0.7480
Epoch 8/10
100/100 [==============================] - 49s 494ms/step - loss: 0.5421 - acc: 0.7313 - val_loss: 0.5390 - val_acc: 0.7400
Epoch 9/10
100/100 [==============================] - 47s 469ms/step - loss: 0.5169 - acc: 0.7409 - val_loss: 0.5364 - val_acc: 0.7390
Epoch 10/10
100/100 [==============================] - 47s 471ms/step - loss: 0.5126 - acc: 0.7403 - val_loss: 0.5207 - val_acc: 0.7450

Increase epochs and steps per epoch for greater accuracy, requiring more processing time.
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(activation = "relu", units = 128))
classifier.add(Dense(activation = "sigmoid", units = 1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./../dataset/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('./../dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 10,
                         verbose = 2,
                         validation_data = test_set)
