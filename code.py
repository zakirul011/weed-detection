import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, Callback

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# Directories
train_dir = '/kaggle/input/weed-detection-in-crop-fields/data/train'
test_dir = '/kaggle/input/weed-detection-in-crop-fields/data/test'

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-5

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)

# Feature Extraction with MobileNetV2
# base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Feature Extraction with VGG16
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Feature Extraction with ResNet50
# base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Feature Extraction with InceptionV3
# base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Feature Extraction with DenseNet121
# base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Feature Extraction with EfficientNetB0
# base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(generator, feature_extractor):
    features, labels = [], []
    for inputs, targets in generator:
        features.append(feature_extractor.predict(inputs))
        labels.append(targets)
        if len(features) >= len(generator):
            break
    return np.vstack(features), np.hstack(labels)

train_features, train_labels = extract_features(train_generator, feature_extractor)
val_features, val_labels = extract_features(val_generator, feature_extractor)
test_features, test_labels = extract_features(test_generator, feature_extractor)

train_features_flat = train_features.reshape(train_features.shape[0], -1)
val_features_flat = val_features.reshape(val_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# Custom callback to log learning rate and loss
class LearningRateLossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(self.model.optimizer.learning_rate.numpy())
        self.losses.append(logs.get('loss'))

# Learning rate scheduler function
def lr_scheduler(epoch, lr):
    return lr * 0.95  # Reduce learning rate by 5% each epoch

lr_logger = LearningRateLossLogger()
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)


# custom callback for Precision, Recall, and F1-Score vs Epochs
class MetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        val_predictions = (self.model.predict(val_generator) > 0.5).astype(int).flatten()
        val_labels = val_generator.classes

        precision = precision_score(val_labels, val_predictions)
        recall = recall_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions)

        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)

# Train model with the custom metrics logger
metrics_logger = MetricsLogger()

# Build and Fine-Tune End-to-End Classifier
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Initial training
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Fine-tune
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
fine_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[lr_logger, metrics_logger, lr_scheduler_callback]
)

# Classifiers
classifiers = {
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

for name, clf in classifiers.items():
    clf.fit(train_features_flat, train_labels)

def evaluate(clf, X, y_true, is_keras_model=False):
    if is_keras_model:
        y_pred = (clf.predict(X) > 0.5).astype(int).flatten()
        y_proba = clf.predict(X).flatten()
    else:
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return accuracy, precision, recall, f1, auc


# Evaluate classifiers
true_labels = test_generator.classes
for name, clf in classifiers.items():
    print(f"\n{name} Performance:")
    metrics = evaluate(clf, test_features_flat, true_labels)
    print(f"Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, "
          f"F1: {metrics[3]:.4f}, AUC: {metrics[4]:.4f}")

# Evaluate Model
metrics = evaluate(model, test_generator, true_labels, is_keras_model=True)
print("\nModel Performance:")
print(f"Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, "
      f"F1: {metrics[3]:.4f}, AUC: {metrics[4]:.4f}")

# Visualization

# Plot Training vs Validation Accuracy and Loss
def plot_training(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.show()

# ROC Curve
def plot_auc_roc(y_true, y_proba, title="AUC-ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=test_generator.class_indices.keys(), 
                yticklabels=test_generator.class_indices.keys())
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot Precision-Recall Curve
def plot_precision_recall(y_true, y_proba, title="Precision vs Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}', color='purple')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()

# Plot Learning Rate vs Loss
def plot_LR(title="Learning Rate vs Loss"):
    plt.figure(figsize=(8, 6))
    plt.plot(lr_logger.lrs, lr_logger.losses, marker='o', color='purple')
    plt.title(title)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.xscale('log')
    plt.grid()
    plt.show()

# Plot Precision, Recall, and F1-Score vs Epochs
def plot_metrices_epochs(title="Precision, Recall, and F1-Score vs Epochs"):
    plt.figure(figsize=(8, 6))
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, metrics_logger.precisions, label='Precision', color='blue')
    plt.plot(epochs_range, metrics_logger.recalls, label='Recall', color='green')
    plt.plot(epochs_range, metrics_logger.f1_scores, label='F1-Score', color='red')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()

# Training vs Validation Accuracy and Loss
plot_training(history, "Initial Training")
plot_training(fine_history, "Fine-Tuning")

# Generate and visualize results
y_proba = model.predict(test_generator).flatten()
y_pred = (y_proba > 0.5).astype(int)

# ROC Curve and AUC
plot_auc_roc(true_labels, y_proba)

# Confusion Matrix
plot_confusion_matrix(true_labels, y_pred)

# Precision-Recall Curve
plot_precision_recall(true_labels, y_proba)

# Learning Rate vs Loss
plot_LR()

# Precision, Recall, and F1-Score vs Epochs
plot_metrices_epochs()
