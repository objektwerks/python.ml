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

The test takes about 12+ minutes:
Using TensorFlow backend.
Found 8000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
Epoch 1/10
2018-01-21 15:27:56.977934: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX
 - 79s - loss: 0.6603 - acc: 0.5970 - val_loss: 0.5914 - val_acc: 0.7005
Epoch 2/10
 - 77s - loss: 0.5902 - acc: 0.6815 - val_loss: 0.5421 - val_acc: 0.7360
Epoch 3/10
 - 80s - loss: 0.5582 - acc: 0.7111 - val_loss: 0.5501 - val_acc: 0.7245
Epoch 4/10
 - 71s - loss: 0.5317 - acc: 0.7314 - val_loss: 0.5167 - val_acc: 0.7435
Epoch 5/10
 - 77s - loss: 0.5008 - acc: 0.7576 - val_loss: 0.4741 - val_acc: 0.7755
Epoch 6/10
 - 82s - loss: 0.4731 - acc: 0.7761 - val_loss: 0.4753 - val_acc: 0.7725
Epoch 7/10
 - 79s - loss: 0.4588 - acc: 0.7801 - val_loss: 0.4472 - val_acc: 0.7925
Epoch 8/10
 - 88s - loss: 0.4568 - acc: 0.7792 - val_loss: 0.5146 - val_acc: 0.7540
Epoch 9/10
 - 84s - loss: 0.4324 - acc: 0.7961 - val_loss: 0.4570 - val_acc: 0.7890
Epoch 10/10
 - 82s - loss: 0.4215 - acc: 0.8066 - val_loss: 0.4434 - val_acc: 0.8045
objektwerks:dl objektwerks$

Increase epochs for greater accuracy, requiring more processing time.
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('./../dataset/training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('./../dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
                                            
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(activation = "relu", units = 128))
classifier.add(Dense(activation = "sigmoid", units = 1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 10,
                         verbose = 2,
                         validation_data = test_set)
