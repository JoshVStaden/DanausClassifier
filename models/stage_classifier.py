import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from .callback_config import CALLBACKS

class StageClassifier:
    def __init__(self, im_pix=32, init_file=None) -> None:
        # self.im_pix=im_pix
        self.im_size = (im_pix, im_pix, 3)
        self.species_data = None
        self.stage_data = ['butterfly', 'pupa','larva']
        if init_file is not None:
            self.stage_classifier = load_model(init_file + ("_size_%d.h5" %im_pix))
        else:
            self.create_stage_classifier()

    def load_data(self):
        # 'X_data', 'y_data', 'target_names'
        butterfly_data = np.load("data/butterfly_%d.npz"%self.im_size[0])
        pupa_data = np.load("data/pupa_%d.npz"%self.im_size[0])
        larva_data = np.load("data/larva_%d.npz"%self.im_size[0])
        self.species_data = np.unique(butterfly_data['target_names'])
        return butterfly_data, pupa_data, larva_data

    def get_concatenated_data(self, butterfly_data, pupa_data, larva_data, max_samples,  keys=['X_data', 'y_data', 'stage']):
        ret_data = {}
        for key in keys:
            b = butterfly_data[key]
            p = pupa_data[key]
            l = larva_data[key]
            if b.shape[0] > max_samples:
                b = b[:max_samples,...]
            if p.shape[0] > max_samples:
                p = p[:max_samples,...]
            if l.shape[0] > max_samples:
                l = l[:max_samples,...]
            # ret_data[key] = np.concatenate((b, p, l))
            ret_data[key] = np.concatenate((b, p, l))
        return ret_data


    def get_classes(self, butterfly_data, pupa_data, larva_data, max_samples=1000):
        # y_data => (species, stage)
        dataset = self.get_concatenated_data(butterfly_data, pupa_data, larva_data, max_samples)
        im_data = dataset['X_data']
        target_data = dataset['y_data']
        stage_data = dataset['stage']

        ind = np.where(stage_data != 2)
        stage_data = stage_data[ind]
        # target_data = target_data[ind]
        im_data = im_data[ind[0],...]

        X_train, X_test, y_train_stage, y_test_stage = train_test_split(im_data, stage_data, test_size=0.3)
        X_train, X_val, y_train_stage, y_val_stage = train_test_split(X_train, y_train_stage, test_size=0.3)

        train_data_gen = ImageDataGenerator(rotation_range=40,
                                    # width_shift_range=0.2,
                                    # height_shift_range=0.2,
                                    shear_range=0.2,
                                    # zoom in from 50% to zoom out to 50%
                                    zoom_range=[0.2,1.2]
                                    # horizontal_flip=True
                                    # vertical_flip=True,
                                    # change the brightness from 20% darker to 20% lighter
                                    # brightness_range=[0.2,1.2]
                                    )
        val_data_gen = ImageDataGenerator()

        train_stage = train_data_gen.flow(X_train, y=to_categorical(y_train_stage))
        val_stage = val_data_gen.flow(X_val, y=to_categorical(y_val_stage))

        return train_stage, val_stage, X_test, y_test_stage


    def create_stage_classifier(self, chosen_model='vgg16'):
        model = Sequential()
        conv_models = {"vgg16": VGG16}
        for l in conv_models[chosen_model](weights="imagenet", include_top=False, input_shape=self.im_size).layers:
            l.trainable = False
            model.add(l)
        model.add(Flatten(name='flatten_layer'))
        model.add(Dense(4096))
        model.add(LeakyReLU())
        model.add(Dense(4096))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        self.stage_classifier = model

    def train_stage_classifier(self, X_train, X_val, batch_size=256, transfer_epochs=150, total_epochs=100000):

        self.stage_classifier.fit(X_train, validation_data=X_val,
                                  batch_size=batch_size,
                                  epochs=transfer_epochs,
                                   callbacks=CALLBACKS)
        for l in self.stage_classifier.layers:
            if l.name == 'flatten_layer':
                break
            l.trainable = True
        rem_epochs = total_epochs - transfer_epochs
        self.stage_classifier.fit(X_train, validation_data=X_val,
                                  batch_size=batch_size,
                                  epochs=rem_epochs,
                                  callbacks=CALLBACKS)


    def predict(self, X):
        return np.argmax(self.stage_classifier.predict(X), axis=1)

    def create_confusion_matrix(self, X_test, y_test, saveas="training_stage_confusion_matrix"):
        y_pred = np.argmax(self.stage_classifier.predict(X_test), axis=1)
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(conf_mat, annot=True, xticklabels=self.stage_data, yticklabels=self.stage_data, cmap="viridis")
        plt.savefig("%s.png"%saveas)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print("Precision Score: %2f" %prec)
        print("Recall Score: %2f" %rec)

    def save_stage_model(self, name="stage"):
        name = "weights/" +  name + ("_size_%d.h5"%self.im_size[0])
        self.stage_classifier.save(name)
        print("Saved model to file %s"%name)

        

