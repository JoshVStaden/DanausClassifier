from ctypes import ArgumentError

import numpy as np
from numpy.lib.npyio import save

import matplotlib.pyplot as plt

class DanausClassifier:
    def __init__(self, stage_classifier, butterfly_classifier, larva_classifier):
        self.im_size = stage_classifier.im_size
        
        self.stage_classifier = stage_classifier
        self.butterfly_classifier = butterfly_classifier
        self.larva_classifier = larva_classifier

    def load_data(self):
        return self.stage_classifier.load_data()

    def get_classes(self, butterfly_data, pupa_data, larva_data, for_classifier="stage"):
        if for_classifier == "stage":
            _, _, X_test_stage, y_test_stage = self.stage_classifier.get_classes(butterfly_data, pupa_data, larva_data)
            return X_test_stage, y_test_stage
        if for_classifier == "adult":
            popular_classes = [0,2,4,5,6,7]
            _, _, X_test_adult, y_test_adult =  self.butterfly_classifier.get_classes(butterfly_data, popular_classes=popular_classes)
            return X_test_adult, y_test_adult
        if for_classifier == "larva":
            _, _, X_test_larva, y_test_larva =  self.larva_classifier.get_classes(larva_data)
            return X_test_larva, y_test_larva
        raise ArgumentError("for_classifier must be either 'stage', 'adult', or 'larva'")


    def get_confusion_matrices(self, X, y_true, for_classifier="stage"):
        if for_classifier == "stage":
            self.stage_classifier.create_confusion_matrix(X, y_true, saveas="testing_stage_classifier")
        elif for_classifier == "adult":
            self.butterfly_classifier.create_confusion_matrix(X, y_true, saveas="testing_butterfly_classifier")
        elif for_classifier == "larva":
            self.larva_classifier.create_confusion_matrix(X, y_true, saveas="testing_larva_classifier")
        else:
            raise ArgumentError("for_classifier must be either 'stage', 'adult', or 'larva'")

    def save_image_prediction(self, im, stage_result, species_result, filename="image"):
        plt.figure()
        plt.imshow(im)
        plt.title(stage_result + " " + species_result)
        plt.savefig("results/" + filename + ".png")

        


    def predict(self, X, save_images=None):        
        stage_y = self.stage_classifier.predict(X)
        idx_butterfly = np.where(stage_y == 0)[0]
        idx_larva = np.where(stage_y == 1)[0]

        stage_results = np.vectorize(lambda x: "Adult" if (x == 0) else "Larva" if (x == 1) else "Pupa")(stage_y)

        out_predictions = np.ones(X.shape[0], dtype=np.int32) * -1
        if len(idx_butterfly) > 0: out_predictions[idx_butterfly] = self.butterfly_classifier.predict(X[idx_butterfly,...])
        if len(idx_larva) > 0: out_predictions[idx_larva] = self.larva_classifier.predict(X[idx_larva,...])
        species_results = np.empty((X.shape[0],), dtype=np.object)
        if len(idx_butterfly) > 0: species_results[idx_butterfly] = np.vectorize(lambda x: self.butterfly_classifier.species[x])(out_predictions[idx_butterfly])
        if len(idx_larva) > 0: species_results[idx_larva] = np.vectorize(lambda x: self.larva_classifier.species[x])(out_predictions[idx_larva])
        if save_images is not None:
            for i in range(X.shape[0]):
                self.save_image_prediction(X[i,...], stage_results[i], species_results[i], filename="%s_prediction_%d"%(save_images, i))

        return out_predictions, (stage_results, species_results)






