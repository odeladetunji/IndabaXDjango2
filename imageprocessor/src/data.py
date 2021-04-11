from keras_preprocessing.image import ImageDataGenerator

from src import params as cfg


def data_generator(train_dir, test_dir):
    """

    :param train_dir:
    :param test_dir:
    :return: the 3 splits data {train, validation, test}
    """

    train_datagen = ImageDataGenerator(rescale=1. / 255, featurewise_center=True,
                                       featurewise_std_normalization=True, validation_split=0.25,
                                       zoom_range=0.2, shear_range=0.2)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(cfg.img_height, cfg.img_width),
                                                        batch_size=cfg.train_batch_size,
                                                        seed=cfg.random_seed,
                                                        shuffle=False,
                                                        subset='training',
                                                        class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(train_dir,
                                                             target_size=(cfg.img_height, cfg.img_width),
                                                             batch_size=cfg.val_batch_size,
                                                             seed=cfg.random_seed,
                                                             shuffle=False,
                                                             subset='validation',
                                                             class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(cfg.img_height, cfg.img_width),
                                                      batch_size=cfg.test_batch_size,
                                                      seed=cfg.random_seed,
                                                      shuffle=False,
                                                      class_mode='categorical')

    return train_generator, validation_generator, test_generator


