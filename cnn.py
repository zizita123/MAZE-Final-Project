from typing import Optional, Dict, List
import tensorflow as tf
import os

DATA_DIR = "/data/images/"

IMAGE_PATH = os.getcwd() + DATA_DIR

class ImagePot:
    def __init__(self, image_paths: List[str], classname: str, encoding: Optional[List[int]] = None):
        self.image_paths = image_paths
        self.classname = classname
        self.encoding = encoding

    def set_encoding(self, encoding) -> None:
        self.encoding = encoding

    def __str__(self) -> str:
        return "Group of Images of Class {0}: containing {1} file(s)".format(
            self.classname, 
            len(self.image_paths)
        )

    def __repr__(self) -> str:
        return str(self)

def load_images(
        num_image_minimum: int = 100, 
        num_image_limit: Optional[int] = 200
    ) -> Dict[str, Dict[str, object]]: 
    # these are the labels enclosing each folder of images
    sub_dirs = sorted([y for x in os.walk(IMAGE_PATH) if (y := (x[0].split(IMAGE_PATH)[-1]))])

    # assume that data processing script has already un-nested image contents
    good_folders = {}

    for folder in sub_dirs:
        file_names = os.listdir("/".join([IMAGE_PATH, folder]))
        num_images = len(file_names)
        print(f"Classname: {folder} has {num_images} images.")
        if num_images >= num_image_minimum:
            good_folders[folder] = {
                'images': ImagePot([IMAGE_PATH + folder + x for x in file_names[:num_image_limit]], folder), 
                'classname': folder,
                'ohe': None,
            }

    # directory of valid images
    ohe_classes = [([0] * len(good_folders)) for _ in range(len(good_folders))]
    for i, k in enumerate(good_folders):
        encoding = ohe_classes[i]
        # set equal to available classes
        encoding[i] = 1
        good_folders[k]['images'].set_encoding(encoding)
        good_folders[k]['ohe'] = encoding

    return good_folders


def build_cnn(num_classes: int, image_dim: int = 32):
    """
    Placeholder values for now
    """
    input_prep_fn = tf.keras.Seqeuential([
        # color scale
        tf.keras.layers.Rescaling(scale=1 / 255),
        # image resizing to n x n pixels
        tf.keras.layers.Resizing(image_dim, image_dim),
    ])

    input_aug_fn = tf.keras.Seqeuential([
        tf.keras.layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15,
            fill_mode='reflect',
            interpolation='bilinear',
            fill_value=0.0,
        ),
        tf.keras.layers.RandomRotation(
            factor=0.1,
            fill_mode='reflect',
            interpolation='bilinear',
            fill_value=0.0,
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.05,
            width_factor=0.05,
            fill_mode='reflect',
            interpolation='bilinear',
            fill_value=0.0,
        ),
        tf.keras.layers.RandomContrast(
            factor=0.05,
        ),
        tf.keras.layers.GaussianNoise(
            stddev=0.1,
        )
    ])

    cnn_model = tf.keras.Sequential([
        # first conv
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), actuvation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),
        # second conv
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), actuvation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        # ----
        tf.keras.layers.Flatten(),
        # dense layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(256, activation='relu'),
        # classifier
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    cnn_model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    return cnn_model

def train_cnn(X_train, X_test, Y_train, Y_test, model, epochs=10, batch_size=100):
    # train  / test the model
    model.fit(
        X_train, 
        Y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, Y_test)
    )
    return model

if __name__ == "__main__":
    data_directory = load_images()
    from pprint import pprint
    pprint(data_directory)
