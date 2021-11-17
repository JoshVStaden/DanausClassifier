# Danaus Classifier
Two-step classification of Danaus species

## Methodology

### Training:
1. Train classifier to determine whether image is adult or larva Danaus
2. Train classifier to recognize species among adult butterflies
3. Train classifier to recognize species among larva butterflies

### Inference:
1. Determine if species is adult or larva
2. Based on the previous result, use the appropriate classifier

## Step-by-Step

### Config
Modify `config.py` to specify:

- Your dataset path
- Your image size. If image size is 128, this will set images to be 128 x 128
- Batch size

### Data Caching

This software caches image files by saving datasets as `.npz` files. To do this, run:

``python cache_data.py``

This should create 3 files in `data/`:
- `butterfly_x.npz`
- `pupa_x.npz`
- `larva_x.npz`

### Training

To train the models, run:

``python train.py``

### Testing

To test the models, run:

``python test.py``
