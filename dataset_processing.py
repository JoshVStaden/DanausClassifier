from pathlib import Path

import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage import io
from skimage.transform import resize

def add_all_images_in_direc(direc, dimension, i, output_data, is_base_directory=True):
    # flat_data = []
    # images = []
    flat_data, target, target_str = output_data
    for file in direc.iterdir():
        if file.suffix != ".jpeg":
            continue
        img = io.imread(file)
        if len(img.shape) < 3:
            continue
        img_resized = resize(
            img[:, :, :3], dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized)
        target.append(i)
        target_str.append(str(direc).split("\\")[-1 if is_base_directory else -2])
    return (flat_data, target, target_str)



def directory_to_dataframe(container_path, dimension=(32, 32, 3), saveas="train_data"):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir()
               if directory.is_dir()]
    butterfly_data = ([], [], [])
    pupa_data = ([], [], [])
    larva_data = ([], [], [])
    total_folders = len(folders)
    for i, direc in enumerate(folders): #species
        print("Folder %d/%d: Adding Butterflies" %(i + 1, total_folders))
        butterfly_data = add_all_images_in_direc(direc, dimension, i, butterfly_data)
        
        print("Folder %d/%d: Adding Pupae" %(i + 1, total_folders))
        pupa_data = add_all_images_in_direc(direc/"pupa", dimension, i, pupa_data, is_base_directory=False)
        
        print("Folder %d/%d: Adding Larvae" %(i + 1, total_folders))
        larva_data = add_all_images_in_direc(direc/"larva", dimension, i, larva_data, is_base_directory=False)
    butterfly_flat_data, butterfly_target, butterfly_target_str = butterfly_data
    butterfly_flat_data = np.array(butterfly_flat_data)
    butterfly_target = np.array(butterfly_target)


    
    pupa_flat_data, pupa_target, pupa_target_str = pupa_data
    pupa_flat_data = np.array(pupa_flat_data)
    pupa_target = np.array(pupa_target)
    total_pupa_classes = np.unique(pupa_target)
    tmp = []
    for i in range(len(total_pupa_classes)):
        pupa_target[np.where(pupa_target == total_pupa_classes[i])] = i
        tmp.append(pupa_target_str[total_pupa_classes[i]])
    pupa_target_str = tmp

    
    larva_flat_data, larva_target, larva_target_str = larva_data
    larva_flat_data = np.array(larva_flat_data)
    larva_target = np.array(larva_target)
    larva_target_str = np.array(larva_target_str)
    total_larva_classes = np.unique(larva_target)
    tmp = []
    for i in range(len(total_larva_classes)):
        curr_class = total_larva_classes[i]
        larva_target[np.where(larva_target == curr_class)] = i
        tmp.append(larva_target_str[curr_class])

    np.savez("data/butterfly_%d.npz"%dimension[0], X_data=butterfly_flat_data, y_data=butterfly_target, target_names=butterfly_target_str, stage=np.zeros((len(butterfly_target_str),)))
    np.savez("data/pupa_%d.npz"%dimension[0], X_data=pupa_flat_data, y_data=pupa_target, target_names=pupa_target_str, stage=np.ones((len(pupa_target_str),)) + 1)
    np.savez("data/larva_%d.npz"%dimension[0], X_data=larva_flat_data, y_data=larva_target, target_names=larva_target_str, stage=np.ones((len(larva_target_str),)) )
    print("Done")