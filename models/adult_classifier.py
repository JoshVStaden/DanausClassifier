import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .callback_config import CALLBACKS


class ButterflyClassifier:
    def __init__(self, im_pix=32, init_file=None) -> None:
        # self.im_pix=im_pix
        self.im_size = (im_pix, im_pix, 3)
        self.species_data = None
        butterfly_data = np.load("data/butterfly_%d.npz"%self.im_size[0])
        self.species = np.unique(butterfly_data['target_names'])
        self.create_butterfly_classifier()

    def create_butterfly_classifier(self, chosen_model='vgg16'):
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
        model.add(Dropout(0.2))
        model.add(Dense(len(self.species), activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        self.butterfly_classifier = model


    def load_data(self):
        # 'X_data', 'y_data', 'target_names'
        butterfly_data = np.load("data/butterfly_%d.npz"%self.im_size[0])
        return butterfly_data


    def get_concatenated_data(self, butterfly_data,  keys=['X_data', 'y_data', 'stage']):
        ret_data = {}
        for key in keys:
            b = butterfly_data[key]
            ret_data[key] = b
        return ret_data

    def clip_classes(self, target_data, max_samples):
        classes = np.unique(target_data)
        new_im_data = np.empty((0,) + self.im_size)
        new_target_data = np.empty((0,))
        ind_arr = []
        for c in classes:
            ind = np.where(target_data == c)[0][:max_samples]
            ind_arr.extend(ind)
            new_target_data = np.concatenate((new_target_data, target_data[ind]))
        return ind_arr, new_target_data



    def get_classes(self, butterfly_data, popular_classes=None, max_samples=1200):
        # y_data => (species, stage)
        if popular_classes is None:
            popular_classes = [i for i in range(len(self.species))]
        dataset = self.get_concatenated_data(butterfly_data)
        # im_data = 
        target_data = dataset['y_data']

        ind = np.where(np.isin(target_data, popular_classes))
        target_data = target_data[ind]
        self.species = self.species[popular_classes]
        

        # Re-label to ensure that classes are indexed 0-5
        for i in range(len(popular_classes)):
            target_data[np.where(target_data == popular_classes[i])] = i

        ind_clip, target_data = self.clip_classes(target_data, max_samples)
        im_data = dataset['X_data'][ind[0]][ind_clip]


        X_train, X_test, y_train_species, y_test_species= train_test_split(im_data, target_data, test_size=0.3)
        X_train, X_val, y_train_species, y_val_species = train_test_split(X_train, y_train_species, test_size=0.3)

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

        train_species = train_data_gen.flow(X_train, y=to_categorical(y_train_species))
        val_species = val_data_gen.flow(X_val, y=to_categorical(y_val_species))
        self.create_butterfly_classifier()

        return train_species, val_species, X_test, y_test_species



    def train_butterfly_classifier(self, X_train, X_val, batch_size=256, transfer_epochs=150, total_epochs=1000000):
        self.butterfly_classifier.fit(X_train, validation_data=X_val,
                                  batch_size=batch_size,
                                  epochs=transfer_epochs,
                                   callbacks=CALLBACKS)
        for l in self.butterfly_classifier.layers:
            if l.name == 'flatten_layer':
                break
            l.trainable = True
        rem_epochs = total_epochs - transfer_epochs
        self.butterfly_classifier.fit(X_train, validation_data=X_val,
                                  batch_size=batch_size,
                                  epochs=rem_epochs,
                                  callbacks=CALLBACKS)


    def create_confusion_matrix(self, X_test, y_test, saveas="training_butterfly_confusion_matrix"):
        y_pred = self.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(conf_mat, annot=True, xticklabels=self.species, yticklabels=self.species, cmap="viridis")
        plt.savefig("%s.png"%saveas)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        print("Precision Score: %2f" %prec)
        print("Recall Score: %2f" %rec)



    def predict(self, X):
        return np.argmax(self.butterfly_classifier.predict(X), axis=1)


    def save_butterfly_model(self, name="butterfly"):
        name = "weights/" +  name + ("_size_%d.h5"%self.im_size[0])
        self.butterfly_classifier.save(name)
        print("Saved model to file %s"%name)
