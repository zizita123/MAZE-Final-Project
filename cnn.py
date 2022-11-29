from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import os

import tensorflow as tf
import cv2
import numpy as np

DATA_DIR = "/data/images/"

IMAGE_PATH = os.getcwd() + DATA_DIR

T_TEST_TRAIN = Tuple[np.array, np.array, np.array, np.array]

class ImagePot:
    # parsed from abbreviations.csv
    CLASSNAME_LOOKUP = {
        'ABE': 'Abnormal eosinophil',
        'ART': 'Artefact',
        'BAS': 'Basophil',
        'BLA': 'Blast',
        'EBO': 'Erythroblast',
        'EOS': 'Eosinophil',
        'FGC': 'Faggott cell',
        'HAC': 'Hairy cell',
        'KSC': 'Smudge cell',
        'LYI': 'Immature lymphocyte',
        'LYT': 'Lymphocyte',
        'MMZ': 'Metamyelocyte',
        'MON': 'Monocyte',
        'MYB': 'Myelocyte',
        'NGB': 'Band neutrophil',
        'NGS': 'Segmented neutrophil',
        'NIF': 'Not identifiable',
        'OTH': 'Other cell',
        'PEB': 'Proerythroblast',
        'PLM': 'Plasma cell',
        'PMO': 'Promyelocyte',
    }

    def __init__(self, image_paths: List[str], classname: str, encoding: Optional[List[int]] = None) -> ImagePot:
        self.image_paths = image_paths
        self.classname = classname
        self.encoding = encoding
        self.full_name = self.CLASSNAME_LOOKUP.get(classname, "?")

    def set_encoding(self, encoding) -> None:
        self.encoding = encoding

    def get_test_train_split(self, split: float = 0.8) -> T_TEST_TRAIN:
        n = len(self.image_paths)
        div = int(n * split)

        labels = [self.encoding] * n

        # split them up
        train_labels, test_labels = np.array(labels[:div]), np.array(labels[div:])
        train, test = self.image_paths[:div], self.image_paths[div:]
        
        # load the images
        train = np.array([cv2.imread(x).astype(np.float32) for x in train])
        test = np.array([cv2.imread(x).astype(np.float32) for x in test])

        # convention in our assignments
        X0, X1 = train, test
        Y0, Y1 = train_labels, test_labels
        return X0, Y0, X1, Y1

    def __str__(self) -> str:
        return "Image Pot containing Class {0} ({1}), {2} file(s)".format(
            self.classname, 
            self.full_name,
            len(self.image_paths),
        )

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.image_paths)

def load_images(
        num_image_minimum: int = 100, 
        num_image_limit: Optional[int] = 200,
        print_files_on_disk: bool = False,
        load_classes: List[str] = None,
    ) -> Dict[str, Dict[str, object]]: 

    # these are the labels enclosing each folder of images
    sub_dirs = sorted([y for x in os.walk(IMAGE_PATH) if (y := (x[0].split(IMAGE_PATH)[-1]))])

    load_classes = load_classes or sub_dirs

    # ....
    sub_dirs = sorted(load_classes)

    # assume that data processing script has already un-nested image contents
    good_folders = {}

    for folder in sub_dirs:
        file_names = os.listdir("/".join([IMAGE_PATH, folder]))
        num_images = len(file_names)
        if print_files_on_disk:
            print(f"Classname: {folder} has {num_images} images.")

        if num_images >= num_image_minimum:
            good_folders[folder] = {
                'pot': ImagePot([IMAGE_PATH + folder + "/" + x for x in file_names[:num_image_limit]], folder), 
                'classname': folder,
                'ohe': None,
            }
        else:
            print(f"WARNING: Classname: {folder} had fewer than {num_image_minimum} images. Had {num_images}")

    # directory of valid images
    ohe_classes = [([0] * len(good_folders)) for _ in range(len(good_folders))]

    for i, k in enumerate(good_folders):
        encoding = ohe_classes[i]
        # set equal to available classes
        encoding[i] = 1
        good_folders[k]['pot'].set_encoding(encoding)

        # simplify repo structure
        good_folders[k] = good_folders[k]['pot']

    return good_folders

def get_nn_data(
        data_repo: Dict, 
        load_classes: List[str] = None,
        test_train_split: float = 0.8, 
        shuffle: bool = False
) -> T_TEST_TRAIN:
    # default to all classes
    load_classes = load_classes or list(data_repo.keys())

    # train
    X0 = np.array([])
    Y0 = np.array([])

    # test
    X1 = np.array([])
    Y1 = np.array([])
    
    total_classes = len(load_classes)
    for i, x in enumerate(data_repo.items()):
        class_name, pot = x
        
        if class_name not in load_classes:
            # skip it
            continue
            
        total_images = len(pot)
        print("Loading {} images in {} ({}/{})".format(
            total_images, 
            class_name, 
            i + 1, 
            total_classes
        ))

        x0, y0, x1, y1 = pot.get_test_train_split(test_train_split)
        # TRAIN
        X0 = x0 if X0.shape[0] == 0 else np.append(X0, x0, axis=0)
        Y0 = y0 if Y0.shape[0] == 0 else np.append(Y0, y0, axis=0)

        # TEST
        X1 = x1 if X1.shape[0] == 0 else np.append(X1, x1, axis=0)
        Y1 = y1 if Y1.shape[0] == 0 else np.append(Y1, y1, axis=0)

    print("loaded the following: X0: {0}, Y0: {1}, X1: {2}, Y1: {3}". format(
        X0.shape,
        Y0.shape,
        X1.shape,
        Y1.shape,
    ))

    X0, Y0 = shuffle_data_and_labels(X0, Y0)
    X1, Y1 = shuffle_data_and_labels(X1, Y1)

    return X0, Y0, X1, Y1

def shuffle_data_and_labels(data: np.array, labels: np.array) -> Tuple[np.array, np.array]:
    x = data.shape[0]
    y = labels.shape[0]

    if x != y:
        raise ValueError("Mismatch between number of data and number of labels")

    idx = tf.random.shuffle(tf.range(start=0, limit=x, dtype=tf.int32))
    s_d = tf.gather(data, idx)
    s_l = tf.gather(labels, idx)
    return s_d, s_l

def load_data_for_training(min_images: int, max_images: int, load_classes: List[str]) -> T_TEST_TRAIN:
    images = load_images(
        num_image_minimum=min_images,
        num_image_limit=max_images, 
        load_classes=load_classes,
    )
    return get_nn_data(images, load_classes, 0.8, shuffle=True)


def build_cnn(num_classes: int, image_dim: int = 32):
    """
    Placeholder values for now
    """
    input_prep_fn = tf.keras.Sequential([
        # color scale
        tf.keras.layers.Rescaling(scale=1 / 255),
        # image resizing to n x n pixels
        tf.keras.layers.Resizing(image_dim, image_dim),
    ])

    input_aug_fn = tf.keras.Sequential([
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
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),
        # second conv
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        # second conv
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.1),
        # ----
        tf.keras.layers.Flatten(),
        # dense layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        # classifier
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    cnn_model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    print("Built CNN model...")
    return cnn_model

def train_cnn(X_train, X_test, Y_train, Y_test, model, epochs=10, batch_size=100):
    # train  / test the model
    model.fit(
        X_train, 
        X_test, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(Y_train, Y_test)
    )
    return model

if __name__ == "__main__":
    # TODO: fix the data loader, it doesn't actually filter stuff out with fewer than n images
    load_classes = ["ART", "BAS", "BLA", "EBO"]
    X0, Y0, X1, Y1 = load_data_for_training(
        min_images=400,
        max_images=400,
        load_classes=load_classes,
    )
    
    model = build_cnn(len(load_classes), image_dim=250)
    train_cnn(X0, Y0, X1, Y1, model)

