

# ================================================================
# Author: Luis Mateo Sanchez Loaiza
# File: cGan.py
# Purpose: Train a Conditional GAN (cGAN) on the Face Mask Dataset
#          to generate class-specific synthetic images for
#          data augmentation in an unsupervised learning experiment.
# ================================================================

import os
from pathlib import Path
from typing import Dict, List, Tuple

import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


# ---------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------
IMAGE_SIZE = 32
CHANNELS = 3
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 80
GENERATOR_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0001
BETA_1 = 0.5
NUM_EXAMPLES_TO_GENERATE = 16
OUTPUT_DIR = "results_cgan"
SEED = 42
REAL_LABEL_SMOOTHING = 0.9
FAKE_LABEL_VALUE = 0.0
LABEL_FLIP_RATE = 0.05
INSTANCE_NOISE_STDDEV = 0.05
GENERATOR_UPDATES_PER_STEP = 2
KERNEL_INIT = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# ---------------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------------
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------
# Helper function: find the image directory that contains the
# class folders "with_mask" and "without_mask"
# ---------------------------------------------------------------

def find_image_root(dataset_root: str) -> Path:
    dataset_path = Path(dataset_root)

    # We search recursively because KaggleHub may place the dataset
    # inside one or more nested folders depending on the download.
    for current_path, dirnames, _ in os.walk(dataset_path):
        dirname_set = set(dirnames)
        if {"with_mask", "without_mask"}.issubset(dirname_set):
            return Path(current_path)

    raise FileNotFoundError(
        "Could not find class folders 'with_mask' and 'without_mask' inside the dataset."
    )


# ---------------------------------------------------------------
# Helper function: collect image file paths and numeric labels
# ---------------------------------------------------------------

def collect_image_paths(image_root: Path, class_names: List[str]) -> Tuple[List[str], List[int]]:
    image_paths: List[str] = []
    labels: List[int] = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for label_index, class_name in enumerate(class_names):
        class_dir = image_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                image_paths.append(str(file_path))
                labels.append(label_index)

    if not image_paths:
        raise ValueError("No images were found in the dataset folders.")

    return image_paths, labels


# ---------------------------------------------------------------
# Helper function: preprocess images into the range [-1, 1]
# This range is commonly used with tanh in the generator output.
# ---------------------------------------------------------------

def preprocess_image(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=CHANNELS, expand_animations=False)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)

    # Scale pixel values from [0, 255] to [-1, 1]
    image = (image / 127.5) - 1.0

    return image, label


# ---------------------------------------------------------------
# Build a tf.data pipeline for efficient training
# ---------------------------------------------------------------


def build_dataset(image_paths: List[str], labels: List[int]) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------------------------------------------------------
# Apply the same simple augmentation to real and fake images so
# the discriminator cannot exploit an augmentation mismatch.
# ---------------------------------------------------------------
def augment_images(images: tf.Tensor) -> tf.Tensor:
    images = tf.image.random_flip_left_right(images)
    return images


def add_instance_noise(images: tf.Tensor, stddev: float) -> tf.Tensor:
    noise = tf.random.normal(tf.shape(images), mean=0.0, stddev=stddev)
    images = images + noise
    images = tf.clip_by_value(images, -1.0, 1.0)
    return images


# ---------------------------------------------------------------
# Build the conditional generator
# The generator receives noise + class label and produces an image.
# ---------------------------------------------------------------

def build_generator(num_classes: int) -> keras.Model:
    noise_input = layers.Input(shape=(LATENT_DIM,), name="noise_input")
    label_input = layers.Input(shape=(1,), dtype="int32", name="label_input")

    # Convert the label into a dense vector representation
    label_embedding = layers.Embedding(num_classes, LATENT_DIM)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    # Combine noise and class information
    x = layers.Concatenate()([noise_input, label_embedding])

    # Project the combined vector into a small spatial feature map
    x = layers.Dense(4 * 4 * 256, use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4, 4, 256))(x)

    # Upsample with resize-convolution blocks to reduce checkerboard artifacts
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False, kernel_initializer=KERNEL_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Final RGB image in the range [-1, 1]
    output_image = layers.Conv2D(
        CHANNELS,
        kernel_size=5,
        padding="same",
        activation="tanh",
        kernel_initializer=KERNEL_INIT,
        name="generated_image",
    )(x)

    return keras.Model([noise_input, label_input], output_image, name="conditional_generator")


# ---------------------------------------------------------------
# Build the conditional discriminator
# The discriminator receives an image + class label and predicts
# whether the image is real or fake for that class.
# ---------------------------------------------------------------

def build_discriminator(num_classes: int) -> keras.Model:
    image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="image_input")
    label_input = layers.Input(shape=(1,), dtype="int32", name="label_input")

    # Convert the label into a spatial map so it can be merged with the image
    label_embedding = layers.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(label_embedding)

    # Concatenate image channels with the class-condition map
    x = layers.Concatenate(axis=-1)([image_input, label_embedding])

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=KERNEL_INIT)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=KERNEL_INIT)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=KERNEL_INIT)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, kernel_initializer=KERNEL_INIT, name="real_or_fake_logit")(x)

    return keras.Model([image_input, label_input], output, name="conditional_discriminator")


# ---------------------------------------------------------------
# Custom cGAN model
# This class defines how one training step updates both the
# discriminator and the generator.
# ---------------------------------------------------------------
class ConditionalGAN(keras.Model):
    def __init__(
        self,
        discriminator: keras.Model,
        generator: keras.Model,
        latent_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, d_optimizer: keras.optimizers.Optimizer, g_optimizer: keras.optimizers.Optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def sample_random_labels(self, batch_size: tf.Tensor) -> tf.Tensor:
        return tf.random.uniform(
            shape=(batch_size, 1), minval=0, maxval=self.num_classes, dtype=tf.int32
        )

    def smooth_and_flip_targets(
        self, real_targets: tf.Tensor, fake_targets: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if LABEL_FLIP_RATE <= 0.0:
            return real_targets, fake_targets

        real_flip_mask = tf.random.uniform(tf.shape(real_targets)) < LABEL_FLIP_RATE
        fake_flip_mask = tf.random.uniform(tf.shape(fake_targets)) < LABEL_FLIP_RATE

        real_targets = tf.where(real_flip_mask, FAKE_LABEL_VALUE, real_targets)
        fake_targets = tf.where(fake_flip_mask, REAL_LABEL_SMOOTHING, fake_targets)
        return real_targets, fake_targets

    def train_step(self, data):
        real_images, real_labels = data
        real_images = augment_images(real_images)
        batch_size = tf.shape(real_images)[0]

        # -------------------------------------------------------
        # Train the discriminator
        # -------------------------------------------------------
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator([random_latent_vectors, real_labels], training=True)
        fake_images = augment_images(fake_images)

        real_images = add_instance_noise(real_images, INSTANCE_NOISE_STDDEV)
        fake_images = add_instance_noise(fake_images, INSTANCE_NOISE_STDDEV)

        combined_images = tf.concat([real_images, fake_images], axis=0)
        combined_labels = tf.concat([real_labels, real_labels], axis=0)

        real_targets = tf.ones((batch_size, 1)) * REAL_LABEL_SMOOTHING
        fake_targets = tf.ones((batch_size, 1)) * FAKE_LABEL_VALUE
        real_targets, fake_targets = self.smooth_and_flip_targets(real_targets, fake_targets)
        discriminator_targets = tf.concat([real_targets, fake_targets], axis=0)

        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_labels], training=True)
            d_loss = self.loss_fn(discriminator_targets, predictions)

        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_weights))

        # -------------------------------------------------------
        # Train the generator
        # -------------------------------------------------------
        g_loss_total = 0.0

        for _ in range(GENERATOR_UPDATES_PER_STEP):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_generator_labels = self.sample_random_labels(batch_size)
            misleading_targets = tf.ones((batch_size, 1)) * REAL_LABEL_SMOOTHING

            with tf.GradientTape() as tape:
                generated_images = self.generator(
                    [random_latent_vectors, random_generator_labels], training=True
                )
                predictions = self.discriminator(
                    [generated_images, random_generator_labels], training=False
                )
                g_loss = self.loss_fn(misleading_targets, predictions)

            g_gradients = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_weights))
            g_loss_total += g_loss

        g_loss = g_loss_total / tf.cast(GENERATOR_UPDATES_PER_STEP, tf.float32)

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


# ---------------------------------------------------------------
# Callback: save generator and discriminator weights after each
# epoch. We use a custom callback because ModelCheckpoint expects
# a built model with a standard forward pass, while our cGAN uses
# a custom train_step and we mainly want to save submodels.
# ---------------------------------------------------------------
class WeightsCheckpoint(keras.callbacks.Callback):
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        generator_path = self.output_dir / f"generator_epoch_{epoch + 1:02d}.weights.h5"
        discriminator_path = self.output_dir / f"discriminator_epoch_{epoch + 1:02d}.weights.h5"

        self.model.generator.save_weights(str(generator_path))
        self.model.discriminator.save_weights(str(discriminator_path))

        print(f"Saved generator checkpoint to: {generator_path}")
        print(f"Saved discriminator checkpoint to: {discriminator_path}")
# ---------------------------------------------------------------
# Callback: save image samples after selected epochs so we can
# visually inspect how the generator improves over time.
# ---------------------------------------------------------------
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_images: int, latent_dim: int, class_names: List[str], output_dir: str):
        super().__init__()
        self.num_images = num_images
        self.latent_dim = latent_dim
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # We create a fixed noise vector so we can compare progress
        # across epochs using the same random inputs.
        self.seed_noise = tf.random.normal(shape=(num_images, latent_dim), seed=SEED)
        half = num_images // 2

        # Half the images will be "with_mask" and half "without_mask"
        label_values = [0] * half + [1] * (num_images - half)
        self.seed_labels = tf.constant(np.array(label_values).reshape(-1, 1), dtype=tf.int32)

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator([self.seed_noise, self.seed_labels], training=False)
        generated_images = (generated_images * 127.5) + 127.5
        generated_images = tf.clip_by_value(generated_images, 0, 255)
        generated_images = tf.cast(generated_images, tf.uint8).numpy()

        rows = 2
        cols = self.num_images // rows
        canvas = Image.new("RGB", (cols * IMAGE_SIZE, rows * IMAGE_SIZE), color=(255, 255, 255))

        for index, image_array in enumerate(generated_images):
            image = Image.fromarray(image_array)
            x_offset = (index % cols) * IMAGE_SIZE
            y_offset = (index // cols) * IMAGE_SIZE
            canvas.paste(image, (x_offset, y_offset))

        output_path = self.output_dir / f"epoch_{epoch + 1:03d}.png"
        canvas.save(output_path)
        print(f"Saved generated sample grid to: {output_path}")


# ---------------------------------------------------------------
# Helper function: generate synthetic images after training.
# These images can later be added to the supervised training set.
# ---------------------------------------------------------------

def generate_synthetic_images(
    generator: keras.Model,
    class_names: List[str],
    images_per_class: int,
    output_dir: str,
) -> Dict[str, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, int] = {}

    for class_index, class_name in enumerate(class_names):
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        noise = tf.random.normal(shape=(images_per_class, LATENT_DIM))
        labels = tf.constant(np.full((images_per_class, 1), class_index), dtype=tf.int32)
        generated_images = generator([noise, labels], training=False)

        # Convert the generated images from [-1, 1] back to [0, 255]
        generated_images = (generated_images * 127.5) + 127.5
        generated_images = tf.clip_by_value(generated_images, 0, 255)
        generated_images = tf.cast(generated_images, tf.uint8).numpy()

        for index, image_array in enumerate(generated_images):
            image = Image.fromarray(image_array)
            image.save(class_dir / f"synthetic_{class_name}_{index:04d}.png")

        summary[class_name] = images_per_class

    return summary


# ---------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------

def main() -> None:
    # -----------------------------------------------------------
    # Download the dataset from KaggleHub
    # These two lines were requested exactly in this format.
    # -----------------------------------------------------------
    dataset_root = kagglehub.dataset_download("omkargurav/face-mask-dataset")
    print("Path to dataset files:", dataset_root)

    # Define the class names explicitly so label 0 and 1 are consistent
    class_names = ["with_mask", "without_mask"]

    # Find the correct image root inside the downloaded dataset folder
    image_root = find_image_root(dataset_root)
    print(f"Image root found at: {image_root}")

    # Collect all image file paths and labels
    image_paths, labels = collect_image_paths(image_root, class_names)
    print(f"Total images found: {len(image_paths)}")

    class_distribution = {
        class_name: labels.count(index) for index, class_name in enumerate(class_names)
    }
    print("Class distribution:", class_distribution)

    # Build the tf.data pipeline
    dataset = build_dataset(image_paths, labels)

    # Build generator and discriminator models
    generator = build_generator(num_classes=len(class_names))
    discriminator = build_discriminator(num_classes=len(class_names))

    # Print model summaries to help with documentation and debugging
    generator.summary()
    discriminator.summary()

    # Create and compile the cGAN model
    cgan = ConditionalGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=LATENT_DIM,
        num_classes=len(class_names),
    )
    cgan.compile(
        d_optimizer=keras.optimizers.Adam(
            learning_rate=DISCRIMINATOR_LEARNING_RATE, beta_1=BETA_1
        ),
        g_optimizer=keras.optimizers.Adam(
            learning_rate=GENERATOR_LEARNING_RATE, beta_1=BETA_1
        ),
    )

    # Create output directories
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(OUTPUT_DIR) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save generated image grids during training
    monitor = GANMonitor(
        num_images=NUM_EXAMPLES_TO_GENERATE,
        latent_dim=LATENT_DIM,
        class_names=class_names,
        output_dir=str(Path(OUTPUT_DIR) / "training_samples"),
    )

    # Save generator and discriminator checkpoints after each epoch
    checkpoint_callback = WeightsCheckpoint(output_dir=str(checkpoints_dir))

    # Train the cGAN
    cgan.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[monitor, checkpoint_callback],
    )

    # Save final generator and discriminator weights
    generator.save_weights(str(Path(OUTPUT_DIR) / "generator_final.weights.h5"))
    discriminator.save_weights(str(Path(OUTPUT_DIR) / "discriminator_final.weights.h5"))

    # Generate synthetic images for data augmentation after training
    synthetic_summary = generate_synthetic_images(
        generator=generator,
        class_names=class_names,
        images_per_class=200,
        output_dir=str(Path(OUTPUT_DIR) / "synthetic_dataset"),
    )

    print("Synthetic image generation complete.")
    print("Synthetic images created per class:", synthetic_summary)
    print(f"All outputs were saved inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
