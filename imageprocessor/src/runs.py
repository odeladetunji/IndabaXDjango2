import os
import math
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.models import load_model
import src.params as cfg
from src.models import XceptiponModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def train_validate(train_generator, validation_generator):
    """ Runs the classification model on a train and validation data

    :param train_features: DNN extracted features path for training set (.npy)
    :param valid_features: DNN extracted features path for validation set (.npy)
    :param train_generator:
    :param validation_generator:
    :return: history of the run saved in features directory
    """
 
    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)
    predict_size_train = int(math.ceil(nb_train_samples / cfg.train_batch_size))
    predict_size_validation = int(math.ceil(nb_validation_samples / cfg.val_batch_size))

    


    train_steps = np.ceil(nb_train_samples / cfg.train_batch_size)
    val_steps = np.ceil(nb_validation_samples / cfg.val_batch_size)

    num_classes = len(train_generator.class_indices)

    print("nb_train_samples:", nb_train_samples)
    print("nb_validation_samples:", nb_validation_samples)
    print("\npredict_size_train:", predict_size_train)
    print("predict_size_validation:", predict_size_validation)
    print("\n num_classes:", num_classes)


    # Initialize model
    xception = XceptiponModel(input_shape=cfg.input_shape, num_classes=cfg.num_classes)

    if cfg.model == 'xception_gap':
        model = xception.xception_gap()
    elif cfg.model == 'xception_alone':

        model = xception.xception_alone()
    else:
        print(" Specify a Model in the params.py")

    # optim = Adam(lr=cfg.lr, beta_1=cfg.beta_1, beta_2=cfg.beta_2, amsgrad=True)
    optim = SGD(lr = cfg.lr, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer=optim, loss = "categorical_crossentropy", metrics=["accuracy"])
    
       
    earlystoper = EarlyStopping(monitor="val_loss", patience=3)
    checkpointer = ModelCheckpoint(filepath=cfg.checkpoint_name, monitor='val_loss', save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3,
                              verbose=1, mode='max', min_lr=0.00001)

    callbacks = [earlystoper,  reduce_lr, checkpointer]
                                
    history = model.fit_generator(train_generator, steps_per_epoch=train_steps,
                                validation_data=validation_generator,
                                validation_steps=val_steps,
                                epochs=cfg.epochs, 
                                verbose=1,
                                callbacks=callbacks)


    with open(cfg.features_dir + cfg.result_file, 'w') as f:
        f.write(str(history.history))

    model.save(cfg.checkpoint_name)


def test(model_path, test_generator):

    nb_test_samples = len(test_generator.filenames)
    predict_size_test = int(math.ceil(nb_test_samples / cfg.test_batch_size))

    print("nb_test_samples:", nb_test_samples)
    print("predict_size_test:", predict_size_test)
    # load model
    model = load_model(model_path)
    # get predictions
    preds = model.predict(test_generator, verbose=1)

    test_labels = test_generator.classes
    test_labels = to_categorical(test_labels,
                                 num_classes=cfg.num_classes)

    predictions = [i.argmax() for i in preds]
    y_true = [i.argmax() for i in test_labels]
    cm = confusion_matrix(y_pred=predictions, y_true=y_true)
    accuracy = accuracy_score(y_true=y_true, y_pred=predictions)
    print('Accuracy {}'.format(accuracy))

    return cm, accuracy


