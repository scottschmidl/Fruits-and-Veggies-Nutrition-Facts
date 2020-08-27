from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from naive_bayes_fv import get_X_y_fv
import os

np.random.seed(1337)
def fv_cnn(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() 
     
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     input_shape=input_shape))  # 1st conv. layer                     
    model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 2nd conv. layer
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten())  # necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(32))  # 32 neurons in this layer
    model.add(Activation('relu'))

    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes))  # 2 final nodes (one for each class)
    model.add(Activation('softmax'))  # softmax at end to pick between classes 0-1    
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    batch_size = 5  # number of training samples used at a time to update the weights
    nb_classes = 2    # number of output possibilities: [0 - 1]
    nb_epoch = 5     # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 32, 32   # the size of the fruits/veggies images
    input_shape = (img_rows, img_cols, 1)   # 1 channel image input (grayscale)
    nb_filters = 2    # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4,4)  # convolutional kernel size, slides over image to learn features

    X = []
    y = []
    y_enumerated = []
    all_fru_veg = os.listdir('data/fruits_vegetables')[10:41:27]
    X, y, _ = get_X_y_fv(X, y, all_fru_veg)
    for fruit in y:
        if fruit == 'Tomato':
            y_enumerated.append(1)
        elif fruit == 'Pear':
            y_enumerated.append(0)
    y_enumerated = np.asarray(y_enumerated)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enumerated) 
          
    ## featurize
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    ## conversion/normalization
    X_train = X_train.astype('float32')  # data was uint8 [0-255]
    X_test = X_test.astype('float32')    # data was uint8 [0-255]
    X_train /= 255  # normalizing (scaling from 0 to 1)
    X_test /= 255   # normalizing (scaling from 0 to 1)
    # y_train = to_categorical(y_train, nb_classes) # use when loss = 'categorical entropy'
    # print('y_train: ', y_train[:20])
    # print(y_train.shape)
    # y_test = to_categorical(y_test, nb_classes) # use when loss = 'categorical entropy'
    # print('y_test: ', y_test[:20])
    # print(y_test.shape)
    # print('X_train:\n', X_train[20])
    # print(X_train.shape)
    # print('X_test:\n', X_test[20])
    # print(X_test.shape)
            
    ## run the model
    activations = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']
    model = fv_cnn(nb_filters, kernel_size, input_shape, pool_size)    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, y_test), shuffle=True)    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
