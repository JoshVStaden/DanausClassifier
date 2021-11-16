from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping

CALLBACKS = [
    TensorBoard(),
    ReduceLROnPlateau(patience=10, monitor='val_loss', verbose=True),
    EarlyStopping(patience=30, monitor='val_loss', verbose=True)
]
