from dataset_processing import directory_to_dataframe
from config import IMAGE_SIZE, DATASET_PATH

directory_to_dataframe(DATASET_PATH, dimension=(IMAGE_SIZE,IMAGE_SIZE,3))