# from keras import Sequential
# from keras.src import optimizers
# from keras.src.callbacks import ModelCheckpoint, EarlyStopping
# from keras.src.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout
# from keras.src.legacy.preprocessing.image import ImageDataGenerator
#
# input_shape = (32, 32, 3)
# num_classes = 10
# train_data_dir = 'split_dataset/train'
# validation_data_dir = 'split_dataset/test'
# batch_size = 50  # Increase batch size
# epochs = 10000 # Increase epochs for more training
# model_save_path = 'model/creditcard_improved.keras'
#
# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# validation_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(32, 32),
#     batch_size=batch_size,
#     class_mode='categorical'
# )
#
# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(32, 32),
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=False
# )
#
# # Build the model
# model = Sequential()
#
# # 2 sets of CRP (Convolution, RELU, Pooling) with Dropout
# model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))  # Add dropout
#
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))  # Add dropout
#
# # Fully connected layers with Dropout
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation("relu"))
# model.add(Dropout(0.5))  # Add dropout
#
# # Softmax (for classification)
# model.add(Dense(num_classes))
# model.add(Activation("softmax"))
#
# # Compile the model with a different optimizer
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.Adam(learning_rate=0.003),
#               metrics=['accuracy'])
#
# # Model checkpoints and early stopping
# checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
# callbacks = [earlystop, checkpoint]
#
# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     callbacks=callbacks,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )
#
# # Save the trained model
# model.save(model_save_path)
from keras import Sequential
from keras.src import optimizers
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator

input_shape = (32, 32, 3)
num_classes = 10
train_data_dir = 'split_dataset/train'
validation_data_dir = 'split_dataset/test'
batch_size = 64
epochs = 1000
model_save_path = 'model/creditcard_best.keras'

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Build the model with a deeper architecture inspired by VGG
model = Sequential()

# First block of CRP (Convolution, RELU, Pooling) with Batch Normalization and Dropout
model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Second block of CRP (Convolution, RELU, Pooling) with Batch Normalization and Dropout
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Third block of CRP (Convolution, RELU, Pooling) with Batch Normalization and Dropout
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Fourth block of CRP (Convolution, RELU, Pooling) with Batch Normalization and Dropout
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Fully connected layers with Batch Normalization and Dropout
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# Compile the model with the Adam optimizer and a learning rate scheduler
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.003),
              metrics=['accuracy'])

# Model checkpoints and ea rly stopping

checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
callbacks = [earlystop, checkpoint, reduce_lr]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save(model_save_path)
