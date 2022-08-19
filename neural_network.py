import tensorflow as tf
from constants import TEST_DIRECTORY, TRAIN_DIRECTORY
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from PIL import Image

class Model:

    def __init__(self):
        pass

    def define_generators(self):
        #We create generators in order to avoid put all the images in memory
        train_datagen = ImageDataGenerator(rescale = 1/255)
        test_datagen = ImageDataGenerator(rescale = 1/255, validation_split= 0.2)

        self.train_generator = train_datagen.flow_from_directory(
            TRAIN_DIRECTORY,
            target_size = (80, 80),
            batch_size = 128,
            class_mode = "categorical",
            color_mode = "grayscale",
            subset = "training"
        )

        self.validation_generator = test_datagen.flow_from_directory(
            TEST_DIRECTORY,
            target_size = (80, 80),
            batch_size = 128,
            class_mode = "categorical",
            color_mode = "grayscale",
            subset = "validation"
        )

        self.test_generator = test_datagen.flow_from_directory(
            TEST_DIRECTORY,
            target_size = (80, 80),
            batch_size = 128,
            class_mode = "categorical",
            color_mode = "grayscale"
        )

        self.labels = list(self.train_generator.class_indices.keys()) #We define the labels

    def create_architecture(self):
        self.model = tf.keras.models.Sequential([
                    Conv2D(32,(3,3),1, activation = 'relu', input_shape = (80,80,1)),
                    MaxPooling2D(),
                    Conv2D(64,(3,3),1, activation = 'relu'),
                    MaxPooling2D(),
                    Flatten(),
                    Dense(450, activation = 'relu'),
                    Dense(285, activation = 'relu'),
                    Dense(128, activation = 'relu'),
                    Dense(2, activation = 'softmax')
        ])

        self.model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])

    def training(self):
        #We use a CallBack that save the model which the epoch with the best validation accuracy
        checkpoint_path = "best_model"
        checkpoint_weighs= ModelCheckpoint(
            filepath = checkpoint_path,
            frecuency = "epoch",
            save_weights_only = False,
            monitor = "val_accuracy",
            save_best_only = True,
            verbose = 0

        )

        #Another CallBack that stop the training if we don´t move so much the loss for each epoch
        checkpoint_earlystopping = EarlyStopping(
            monitor = 'loss',
            patience = 2,
            mode = 'auto'
        )


        self.model.fit(
            self.train_generator, 
            epochs= 15, 
            callbacks = [checkpoint_earlystopping, checkpoint_weighs],
            validation_data = self.validation_generator
        )    

    def evaluation(self):
        performance = self.model.evaluate(self.test_generator)
        print("Accuracy: ", performance[1])

    def load_model(self):
        try:
            self.model = load_model("best_model")
            return True
        except:
            print("You should train the model before...")
            return False


