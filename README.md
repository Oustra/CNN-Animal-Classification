# 🐾 CNN Animal Classification

This project demonstrates how to build and train Convolutional Neural Networks (CNNs) to classify animal images.  
The notebook covers the full workflow: dataset preparation, preprocessing, model building, training, evaluation, and prediction.

---

## 📂 Project Structure
- **Dataset**: Images are divided into three directories:
  - `train/` – used for training
  - `validation/` – used for validation during training
  - `test/` – used for final evaluation
- **Notebook**: `CNN_animal_Classification.ipynb`
- **README**: Explains the notebook’s workflow step by step

---

## 🚀 Workflow Overview

### 1. Dataset Loading
- Images are loaded from the dataset directories using Keras’ `image_dataset_from_directory`.
- Data is batched, resized to a fixed size, and shuffled (for training).
- Class labels are automatically inferred from the folder names.

### 2. Preprocessing
- Images are resized to the chosen `IMG_SIZE`.
- Normalization is applied to scale pixel values.
- The datasets are optimized with caching and prefetching to improve training speed.

### 3. Model Architectures
Two approaches are explored in the notebook:

1. **Custom CNN**
   - Sequential model with multiple convolutional + pooling layers.
   - Dropout layers help prevent overfitting.
   - Fully connected (Dense) layers at the end for classification.

2. **Transfer Learning with VGG16**
   - Pretrained VGG16 (trained on ImageNet) is used as a feature extractor.
   - The last classification layer is replaced with a new Dense layer matching the number of animal classes.
   - Early layers can be frozen to keep pretrained weights.

### 4. Training
- Models are trained on the training dataset, with validation data monitored per epoch.
- Training history (loss and accuracy) is logged for later visualization.
- Configurable parameters include `epochs`, `batch_size`, and random seed for reproducibility.

### 5. Evaluation
- After training, the model is evaluated on the test dataset.
- Test loss and accuracy are printed to measure final performance.

### 6. Predictions
- The trained model makes predictions on the test dataset.
- Predictions are returned as class probabilities, which are converted into class indices.
- True labels are compared with predictions to check performance.
- Class indices can be mapped back to readable class names.

---

## 📊 Results & Metrics
- Training and validation accuracy/loss are tracked during training.
- Final evaluation reports test accuracy and loss.
- Predictions can be analyzed further with confusion matrices or classification reports.
