MobileNetV2 Food Mold Detection Model

This project implements a Deep Learning model using MobileNetV2 to classify food images into two categories: spoiled and not_spoiled. The model was trained on a custom dataset collected manually and includes various training enhancements such as seed-ensemble evaluation, heavy data augmentation, and best-model checkpointing.


Overview

The goal of this project is to develop a lightweight and efficient neural network that can detect food spoilage of Indian prepared food from image inputs. The model is suitable for mobile deployment, real-time inference, and low-compute environments due to the efficiency of MobileNetV2.

The full training pipeline includes:

* Data augmentation for generalization
* Multi-seed training for stable accuracy
* Best model selection across seeds
* Evaluation with confusion matrix and classification report
* Manual testing on individual images and test folders



Dataset

This project uses a hybrid dataset composed of:

1. Fresh Food Images (External Source)

A portion of the fresh food images was taken from the publicly available dataset:

Indian Food Images Dataset(Hugging Face): https://huggingface.co/datasets/rajistics/indian_food_images


2. Fresh Food Images (Collected manually)

Additional fresh food images were manually collected using:

* Smartphone cameras

* Real-world food items in natural lighting


3. Spoiled Food Images (AI-Generated + Collected manually)

The spoiled food dataset consists of:

* A portion of real spoiled food images captured manually.

* Additional images AI-generated using modern generative models(Gemini, Perplexity) to enhance variability and increase data volume. 


Dataset Structure

Food_dataset/
│
├── train/
│   ├── not_spoiled/
│   └── spoiled/
│
├── val/
│   ├── not_spoiled/
│   └── spoiled/
│
└── test

* Training Set: Used for model learning
* Validation Set: Used to monitor accuracy per epoch
* Test Set: Used for final model verification


Model Architecture

The base model used is MobileNetV2 (pretrained on ImageNet).

This model has been fully trained and implemented using PyTorch.

Modifications applied:

Final classifier layer replaced with a 2-class output layer.

Trained with extensive augmentations:

  * Random cropping
  * Rotations
  * Horizontal flips
  * Color jitter
  * Random erasing

Loss function: CrossEntropyLoss

Optimizer: Adam with weight decay

Training repeated over multiple seeds: [10, 21, 32, 42, 70, 99]

Best-performing seed’s weights were saved automatically



Training Summary

* Model trained for 15 epochs per seed
* Best model selected based on highest validation accuracy: 0.9545 (seed=21)
* Final saved model: mobilenetv2_food_spoilage_best_seed_21.pth


Evaluation included:

* Classification Report
* Confusion Matrix
* Validation Loss Plot


How to Test the Model

* Load and Use the Saved Model

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("mobilenetv2_food_spoilage_best_seed_21.pth"))
model.eval()

* You can test the model in two ways:

1. Test on the Entire Test Directory

2. Test a Single Image



File Explanation

File                                          Description                                                                           

mobilenetv2_food_spoilage_best_seed_21.pth    Best model weights based on validation accuracy                                       
Dataset                                       Contains a google Drive link for the Food dataset folder                                            
MobileNetV2.ipynb                             Full pipeline including dataset loading, augmentation, training loops, and evaluation 



Final Notes

* The model is lightweight and suitable for deployment.
* Dataset was collected manually for spoiled food detection.
* Augmentation and multi-seed training improve robustness.
* Users can easily extend dataset or retrain the model.

