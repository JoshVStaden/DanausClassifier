from models import DanausClassifier, StageClassifier, ButterflyClassifier, LarvaClassifier
from config import IMAGE_SIZE

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