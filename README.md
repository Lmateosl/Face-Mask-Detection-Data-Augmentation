# 🧠 Face Mask Detection — Data Augmentation Project

This project explores different data augmentation strategies using generative models to improve face mask classification performance.

We focus on generating synthetic images using multiple approaches, including Autoencoders, Conditional GANs (cGAN), and standard GANs, and then evaluating their impact on supervised models such as Convolutional Neural Networks (CNNs) and Neural Networks.

The goal is to understand which generative method produces the most useful synthetic data for improving classification accuracy.

---

## 📂 Project Structure

Each team member contributes a specific component of the pipeline:

- Data Augmentation (Generative Models)
- Supervised Models (CNN / NN)
- Evaluation & Comparison

---

## 🤖 cGAN (Luis Mateo)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Model](https://img.shields.io/badge/Model-cGAN-purple)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

### 📌 Overview

In this section, we implemented a Conditional Generative Adversarial Network (cGAN) to generate synthetic images conditioned on class labels (*with_mask* and *without_mask*).

The model was designed to learn class-specific distributions and generate images that match each category.

---

### ⚙️ Approach

- Used the Face Mask Dataset via KaggleHub
- Implemented a **conditional generator** using noise + label embeddings
- Implemented a **conditional discriminator** using image + label input
- Applied training stabilization techniques:
  - Label smoothing
  - Label flipping
  - Instance noise
  - Batch normalization
  - LeakyReLU activations

---

### 🔍 Challenges

Training the cGAN proved to be difficult due to:

- High instability during training
- Mode collapse (generator producing identical images)
- Limited dataset size per class
- Difficulty learning conditional distributions

---

### 📊 Key Observations

- The model often generated low-quality or repetitive images
- Loss values appeared stable but did not reflect visual quality
- Conditioning on labels increased complexity significantly

---

### 💡 Conclusion

Although cGAN provides more control over generated outputs, it requires more data and careful tuning. In this project, it struggled to produce high-quality images consistently, highlighting the challenges of conditional generative modeling.

---

### ▶️ How to Run

To execute the cGAN model, run the following command from the project root:

```bash
python cGan.py
```

Once executed:

- The model will start training automatically
- Generated images will be saved after each epoch
- A folder named `results_cgan/` will be created

Inside this folder, you will find:

- `training_samples/` → generated images during training
- `checkpoints/` → saved model weights
- `synthetic_dataset/` → final generated images for data augmentation

This allows you to visually evaluate the model performance and reuse the generated images for the classification stage.

---
