import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

test_ds = '/home/flynn/482_project/classifier/dataset/DATASET/TEST/'
train_ds = '/home/flynn/482_project/classifier/dataset/DATASET/TRAIN/'

# Load the pre-trained model
base_model = MobileNetV2(input_shape=(160, 160, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the model
base_model.trainable = False

# Add a classification layer to end
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(2)(x)  # 2 classes: organic and recyclable

# Join the model
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Load the dataset in 80/20 train/test split
train_dataset = image_dataset_from_directory(train_ds,
                                             validation_split=0.2,
                                             subset="training",
                                             seed=1337,
                                             image_size=(160, 160),
                                             batch_size=32)
valid_dataset = image_dataset_from_directory(test_ds,
                                             validation_split=0.2,
                                             subset="validation",
                                             seed=1337,
                                             image_size=(160, 160),
                                             batch_size=32)

# Train
model.fit(train_dataset, validation_data=valid_dataset, epochs=100)
model.save('my_model_2.h5') 