import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models import DanausClassifier, StageClassifier, ButterflyClassifier, LarvaClassifier
from dataset_processing import directory_to_dataframe

IMAGE_SIZE = 128
BATCH_SIZE = 2

# ###########################################################
# # Uncomment to convert the Danaus dataset to a .npz file
# ###########################################################

# directory_to_dataframe("E:\MachineLearning\Datasets\Danaus_images", dimension=(IMAGE_SIZE,IMAGE_SIZE,3))


# ###########################################################


# ##########################################################
# # Training Code For Stage Classifier (Uncomment to train data)
# ##########################################################

# s_class = StageClassifier(im_pix=IMAGE_SIZE)
# butterfly_data, pupa_data, larva_data = s_class.load_data()

# train_stage, val_stage, X_test, y_test_stage = s_class.get_classes(butterfly_data, pupa_data, larva_data)
# s_class.train_stage_classifier(train_stage, val_stage, batch_size=BATCH_SIZE)
# s_class.create_confusion_matrix(X_test, y_test_stage)
# # user_in = input("Save Model? y/n : ")
# # if user_in.lower() != "n":
# #     s_class.save_stage_model()
# s_class.save_stage_model()

# ##########################################################


# ##########################################################
# # Training Code For Adult Butterfly Classifier (Uncomment to train data)
# ##########################################################

# b_class = ButterflyClassifier(im_pix=IMAGE_SIZE, init_file="butterfly")
# butterfly_data = b_class.load_data()
# popular_classes = [0,2,4,5,6,7] # These are the classes with enough samples to train properly



# train_stage, val_stage, X_test, y_test_stage = b_class.get_classes(butterfly_data, popular_classes=popular_classes)
# b_class.train_butterfly_classifier(train_stage, val_stage, batch_size=BATCH_SIZE)
# b_class.create_confusion_matrix(X_test, y_test_stage)
# # user_in = input("Save Model? y/n : ")
# # if user_in.lower() != "n":
# #     b_class.save_butterfly_model()
# b_class.save_butterfly_model()

# ###########################################################


# ###########################################################
# # Training Code For Larva Classifier (Uncomment to train data)
# ###########################################################
# l_class = LarvaClassifier(im_pix=IMAGE_SIZE)
# butterfly_data = l_class.load_data()

# train_stage, val_stage, X_test, y_test_stage = l_class.get_classes(butterfly_data)
# l_class.train_larva_classifier(train_stage, val_stage, batch_size=BATCH_SIZE)
# l_class.create_confusion_matrix(X_test, y_test_stage)
# # user_in = input("Save Model? y/n : ")
# # if user_in.lower() != "n":
# #     l_class.save_larva_model()
# l_class.save_larva_model()
# ###########################################################


stage_class = StageClassifier(im_pix=IMAGE_SIZE, init_file="weights/stage")
b_class = ButterflyClassifier(im_pix=IMAGE_SIZE, init_file="weights/butterfly")
l_class = LarvaClassifier(im_pix=IMAGE_SIZE, init_file="weights/larva")

d_class = DanausClassifier(stage_class, b_class, l_class)

butterfly_data, pupa_data, larva_data = d_class.load_data()

def classification_results(classifier, show_images=True):
    X, y = d_class.get_classes(butterfly_data, pupa_data, larva_data, for_classifier=classifier)
    d_class.get_confusion_matrices(X, y, for_classifier=classifier)
    if show_images:
        _, _ = d_class.predict(X[:10,...], save_images=classifier)

classification_results("adult")
classification_results("stage")
classification_results("larva")
