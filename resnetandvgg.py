
# import the libraries as shown below
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import os
import random
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.applications import ResNet50



# re-size all the images to this
IMAGE_SIZE = [224, 224]
targetsize = (224, 224)
train_path = 'mri dataset/train'
valid_path = 'mri dataset/test'

# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of output classes
folders = glob('mri dataset/train/*')

print("Dataset Loaded")
# our layers - you can add more if you want
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# -------------------- RESNET50 MODEL --------------------

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze layers
for layer in resnet.layers:
    layer.trainable = False

x_res = Flatten()(resnet.output)
prediction_res = Dense(len(folders), activation='softmax')(x_res)

model_resnet = Model(inputs=resnet.input, outputs=prediction_res)

model_resnet.summary()

model_resnet.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Make sure you provide the same target size as initialized for the image size
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=targetsize,
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size=targetsize,
                                            batch_size=32,
                                            class_mode='categorical')

def display_sample_image():
    # Select a random image from the training dataset
    class_folders = os.listdir(train_path)  # Get class names
    random_class = random.choice(class_folders)  # Choose a random class
    class_path = os.path.join(train_path, random_class)  # Path to the selected class
    image_files = os.listdir(class_path)  # Get image files
    random_image = random.choice(image_files)  # Choose a random image
    image_path = os.path.join(class_path, random_image)  # Full image path

    # Load original image
    original_image = cv2.imread(image_path)  # Read image using OpenCV
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Preprocess image (Resize but DO NOT Normalize yet)
    preprocessed_image = cv2.resize(original_image, targetsize)  # Resize

    # Augmentation setup (DO NOT use rescale=1./255 here)
    augmentor = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Convert image to array (keep original pixel values for augmentation)
    image_array = img_to_array(preprocessed_image)
    image_array = np.expand_dims(image_array, axis=0)

    # Generate augmented image (Ensure it's in correct format)
    augmented_image = next(augmentor.flow(image_array, batch_size=1))[0].astype("uint8")

    # Display all images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(preprocessed_image)
    ax[1].set_title("Preprocessed Image (Resized)")
    ax[1].axis("off")

    ax[2].imshow(augmented_image)
    ax[2].set_title("Augmented Image (Shear/Zoom/Flip Applied)")
    ax[2].axis("off")

    plt.show()

# Call the function to display an example image
display_sample_image()


# Add checkpoint to save the best model during training
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_model2.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[checkpoint]
)

r_resnet = model_resnet.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Get true labels
true_labels = test_set.classes

# Get class names
class_names = list(test_set.class_indices.keys())

# Predict probabilities
pred_prob = model.predict(test_set, verbose=1)

# Convert probabilities to predicted labels
pred_labels = np.argmax(pred_prob, axis=1)

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("Classification Report:\n")
report = classification_report(true_labels, pred_labels, target_names=class_names)
print(report)



precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')
acc = accuracy_score(true_labels, pred_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", acc)

# Accuracy graph
plt.figure(figsize=(8,5))
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_graph.png")
plt.show()

# Loss graph
plt.figure(figsize=(8,5))
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_graph.png")
plt.show()

from sklearn.metrics import precision_recall_fscore_support

epoch_precision = []
epoch_recall = []
epoch_f1 = []

for epoch in range(len(r.history['loss'])):
    pred_prob = model.predict(test_set, verbose=0)
    pred_labels = np.argmax(pred_prob, axis=1)

    p, rc, f, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    epoch_precision.append(p)
    epoch_recall.append(rc)
    epoch_f1.append(f)

plt.figure(figsize=(10,5))
plt.plot(epoch_precision, label='Precision')
plt.plot(epoch_recall, label='Recall')
plt.plot(epoch_f1, label='F1-score')
plt.title("Precision, Recall, F1 Score per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.savefig("precision_recall_f1.png")
plt.show()


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_loss')
plt.show()

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_acc')
plt.show()



plt.figure(figsize=(10,6))

plt.plot(r.history['accuracy'], label='VGG16 Train Accuracy')
plt.plot(r.history['val_accuracy'], label='VGG16 Val Accuracy')

plt.plot(r_resnet.history['accuracy'], label='ResNet50 Train Accuracy')
plt.plot(r_resnet.history['val_accuracy'], label='ResNet50 Val Accuracy')

plt.title("VGG16 vs ResNet50 Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_comparison.png")
plt.show()


plt.figure(figsize=(10,6))

plt.plot(r.history['loss'], label='VGG16 Train Loss')
plt.plot(r.history['val_loss'], label='VGG16 Val Loss')

plt.plot(r_resnet.history['loss'], label='ResNet50 Train Loss')
plt.plot(r_resnet.history['val_loss'], label='ResNet50 Val Loss')

plt.title("VGG16 vs ResNet50 Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_comparison.png")
plt.show()


vgg_acc = max(r.history['val_accuracy'])
resnet_acc = max(r_resnet.history['val_accuracy'])

print("Best VGG16 Validation Accuracy:", vgg_acc)
print("Best ResNet50 Validation Accuracy:", resnet_acc)
