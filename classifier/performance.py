import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import precision_score, recall_score, f1_score

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

## REPLACE TO YOUR PATH
test_ds = '/home/flynn/482_project/classifier/dataset/DATASET/TEST/'

loaded_model = tf.keras.models.load_model('my_model_2.h5')

valid_dataset = image_dataset_from_directory(test_ds,
                                             validation_split=0.2,
                                             subset="validation",
                                             seed=1337,
                                             image_size=(160, 160),
                                             batch_size=32)

# Extract true labels from the validation dataset
true_labels = []
for images, labels in valid_dataset:
    true_labels.extend(labels.numpy())

# Get model's predictions
predictions = loaded_model.predict(valid_dataset)
predicted_labels = tf.argmax(predictions, axis=1).numpy()

precision_value = precision_score(true_labels, predicted_labels)
recall_value = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f'Precision: {precision_value}')
print(f'Recall: {recall_value}')
print(f'F1-Score: {f1}')