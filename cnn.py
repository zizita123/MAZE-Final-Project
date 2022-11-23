import tensorflow as tf

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
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), actuvation='relu', padding='same')    
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(.3),
        # second conv
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), actuvation='relu', padding='same')    
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
