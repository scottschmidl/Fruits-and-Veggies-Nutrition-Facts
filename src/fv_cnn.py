from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from class_fruit_veggies_NB import FruitsVeggiesNB
import os
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import pickle

np.random.seed(1337)
def fv_cnn(nb_filters, kernel_size, input_shape, pool_size, activ_func):
    '''Fruits and Veggies Convolution Neural Network'''
    model = Sequential() 
     
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     input_shape=input_shape))  # 1st conv. layer                     
    model.add(Activation(activ_func))  # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid'))  # 2nd conv. layer
    model.add(Activation(activ_func))

    model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten())  # necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(32))  # 32 neurons in this layer
    model.add(Activation(activ_func))

    model.add(Dropout(0.5))  # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes))  # 2 final nodes (one for each class)
    model.add(Activation('softmax'))  # softmax at end to pick between classes 0-1    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def tensor_board():
    '''view tensorflow model after it has trained using tensorboard'''
    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    return tbCallBack

def early_stopping():
    earlyStopping = EarlyStopping(monitor='acc', min_delta=0.001, patience=0, verbose=0, mode='auto')
    return earlyStopping

if __name__ == '__main__':
    ###### turn this into a class ###### 
    batch_size = 20  # number of training samples used at a time to update the weights
    nb_classes = 2    # number of output possibilities: [0 - 1]
    nb_epoch = 10     # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 32, 32   # the size of the fruits/veggies images
    input_shape = (img_rows, img_cols, 3) # 1 channel image input (grayscale)
    nb_filters = 4    # number of convolutional filters to use
    pool_size = (3, 3)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4,4)  # convolutional kernel size, slides over image to learn features
    activ_func = 'linear'

    X = []
    y = []
    y_enumerated = []
    grayscale = False
    edge = False
    all_fru_veg = os.listdir('data/fruits_vegetables')
    fru_veg_class = FruitsVeggiesNB(X, y, all_fru_veg)
    X, y, _ = fru_veg_class.get_X_y_fv(X, y, all_fru_veg, grayscale=grayscale, edge=edge)
    for fruit in y:
        if fruit == 'Tomato':
            y_enumerated.append(1)
        elif fruit == 'Pear':
            y_enumerated.append(0)
    y_enumerated = np.asarray(y_enumerated)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enumerated) 
    
    ## featurize
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

    ## conversion/normalization
    X_train = X_train.astype('float32')  # data was uint8 [0-255]
    X_test = X_test.astype('float32')    # data was uint8 [0-255]
    X_train /= 255  # normalizing (scaling from 0 to 1)
    X_test /= 255   # normalizing (scaling from 0 to 1)
    y_train = to_categorical(y_train, nb_classes) # use when loss = 'categorical entropy'
    print('y_train: ', y_train[:20])
    print(y_train.shape)
    y_test = to_categorical(y_test, nb_classes) # use when loss = 'categorical entropy'
    print('y_test: ', y_test[:20])
    print(y_test.shape)
    print('X_train:\n', X_train[20])
    print(X_train.shape)
    print('X_test:\n', X_test[20])
    print(X_test.shape)
            
    ## run the model
    activations = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']
    model = fv_cnn(nb_filters, kernel_size, input_shape, pool_size, activ_func) 
    # tb = tensor_board()
    # # model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test),
    #             validation_split=0.2, class_weight='auto', shuffle=True, callbacks=[tb])   
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, y_test), shuffle=True)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    filename = 'fv_app/fv_cnn_model.sav'
    model.save(filename)

    ## might need to replace the model.fit from above
    ## after python script run successfully, then in terminal run: tensorboard --logidr ./logs
    # tb = tensor_board()
    # model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2,
    #            class_weight='auto', shuffle=True, callbacks=[tb])

    ## to stop at a convergence critia.
    ## might need to replace the model.fit from above
    # es = early_stopping()
    # model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2,
    #           class_weight='auto', shuffle=True, callbacks=[es])
