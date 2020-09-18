import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np

np.random.seed(1337)
def fv_cnn(nb_filters, kernel_size, input_shape, pool_size, activ_func, tensor=False, easystop=False):
    '''Fruits and Veggies Convolution Neural Network'''
    model = Sequential() 
     
    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     input_shape=input_shape)) # 1st conv. layer                     
    model.add(Activation(activ_func)) # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters,
                     (kernel_size[0], kernel_size[1]),
                     padding='valid')) # 2nd conv. layer
    model.add(Activation(activ_func))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(32)) # 32 neurons in this layer
    model.add(Activation(activ_func))

    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes)) # 2 final nodes (one for each class)
    model.add(Activation('sigmoid')) # softmax at end to pick between classes 0-1    
    
    model.compile(loss='binary_crossentropy',
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
    activations = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']
    batch_size = 32 # number of training samples used at a time to update the weights
    nb_classes = 1 # number of output possibilities: [0 - 1]
    nb_epoch = 10 # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 32, 32 # the size of the fruits/veggies images
    input_shape = (img_rows, img_cols, 3) # 1 channel image input (grayscale)
    nb_filters = 4 # number of convolutional filters to use
    pool_size = (3, 3) # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4,4) # convolutional kernel size, slides over image to learn features
    activ_func = 'relu'

    ### Note:image_dataset_from_directory() will replace ImageDataGenerator in tensorflow 2.3.0
    ### featurize data
    datagen = ImageDataGenerator(
                rotation_range=15,
                shear_range=0.2,
                zoom_range=0.2,
                fill_mode='nearest',
                horizontal_flip=True,
                vertical_flip=True,
                rescale=1/255,
                data_format='channels_last',        
                validation_split=0.2)
    print('datagen', datagen, '\n')
    X_train_generator = datagen.flow_from_directory(
                        directory='data/fruits_vegetables',
                        target_size=(32, 32),
                        color_mode='rgb',
                        class_mode='binary',
                        batch_size=batch_size,
                        shuffle=True,
                        seed=23,
                        subset='training')
    print('X_train_gen', X_train_generator, '\n')
    X_validation_generator = datagen.flow_from_directory(
                            directory='data/fruits_vegetables',
                            target_size=(32, 32),
                            color_mode='rgb',
                            class_mode='binary',
                            batch_size=batch_size,
                            shuffle=True,
                            seed=23,
                            subset='validation')   
    print('X_val_gen', X_validation_generator, '\n')
    ### run the model
    model = fv_cnn(nb_filters, kernel_size, input_shape, pool_size, activ_func)
    ### fit the model
    model.fit(
        x=X_train_generator,
        epochs=nb_epoch,
        verbose=1,
        validation_data=X_validation_generator,
        steps_per_epoch=(4391 // batch_size),
        validation_steps=500,
        validation_batch_size=batch_size,
        validation_freq=1,
        max_queue_size=10,
        workers=1)
    ### evaluate model
    score = model.evaluate(
            x=X_validation_generator,
            batch_size=batch_size,
            verbose=1,
            steps=4391 // batch_size,
            max_queue_size=10,
            workers=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    ### save the model
    filename = '../fv_app/fv_cnn_model.sav'
    model.save(filename)
    ### check model.fit parameters before running
    ### to stop at a convergence critia and get tensorboard, use easystop
    ### after python script run successfully, then in terminal run: tensorboard --logidr ./logs
    tensorboard = True
    easystop = False
    if tensorboard:
        tbCallBack = tensor_board()
        model.fit(X_train_generator, X_validation_generator,
                batch_size=batch_size, epochs=nb_epoch, verbose=1,
                validation_split=0.2, class_weight='auto', shuffle=True,
                callbacks=[tbCallBack])
    if easystop:
        earlyStopping = early_stopping()
        model.fit(X_train_generator, X_validation_generator,
                batch_size=batch_size, epochs=nb_epoch, verbose=1,
                validation_split=0.2, class_weight='auto', shuffle=True,
                callbacks=[earlyStopping])    
  
