import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
from keras import ops

from google.colab import drive
drive.mount('/content/drive')

dataset_path = "/content/drive/MyDrive/dataset_folder"

# Function to load and preprocess an image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Change to png if necessary
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return image

# data
dataset_name = "LoomGen"
dataset_repetitions = 5
num_epochs = 200
image_size = 64

kid_image_size = 75

kid_diffusion_steps = 10  # For KID computation
plot_diffusion_steps = 60  # For visualization

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 3

# optimization
batch_size = 16
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4

def prepare_full_dataset(directory):
    image_paths = [os.path.join(directory, fname)
                  for fname in os.listdir(directory)
                  if fname.endswith(('.jpg', '.png'))]

    full_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    full_dataset = full_dataset.cache()
    full_dataset = full_dataset.repeat(dataset_repetitions)
    full_dataset = full_dataset.shuffle(10 * batch_size)
    full_dataset = full_dataset.batch(batch_size, drop_remainder=True)
    full_dataset = full_dataset.prefetch(tf.data.AUTOTUNE)
    return full_dataset

full_dataset = prepare_full_dataset(dataset_path)

#===========================#
# Loom-GenNet Architecture  #
#===========================#

@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = ops.exp(
        ops.linspace(
            ops.log(embedding_min_frequency),
            ops.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = ops.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=True, scale=True)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply

def SpatialPyramidPooling(x, levels=[1, 2, 4]):
    """Apply SPP to extract multi-scale features."""
    pooled_features = []
    for level in levels:
        pool_size = (max(1, x.shape[1] // level), max(1, x.shape[2] // level))
        pooled = layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)(x)
        # Adjust channels with 1x1 convolution and upsample to original size
        pooled = layers.Conv2D(x.shape[3], 1, padding='same')(pooled)
        pooled = layers.UpSampling2D(size=(x.shape[1] // pooled.shape[1], x.shape[2] // pooled.shape[2]), interpolation='bilinear')(pooled)
        pooled_features.append(pooled)
    return layers.Concatenate()(pooled_features)

def get_network(image_size, widths, block_depth):
    input_image = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # ===================================#
    # Asymmetric Dual Encoder Structure  #
    # ===================================#

    # Encoder Path 1 (3x3 convs with dilated convolutions)
    x_left = layers.Conv2D(widths[0], 3, padding='same')(input_image)
    x_left = layers.Concatenate()([x_left, e])
    x_left = layers.Conv2D(widths[0], 3, padding='same', dilation_rate=2)(x_left)

    # Encoder Path 2 (5x5 convs with dilated convolutions)
    x_bottom = layers.Conv2D(widths[0], 5, padding='same')(input_image)
    x_bottom = layers.Concatenate()([x_bottom, e])
    x_bottom = layers.Conv2D(widths[0], 5, padding='same', dilation_rate=2)(x_bottom)

    skips = []

    # Process both encoders
    for width in widths[:-1]:
        # Encoder 1 (3x3 with increasing dilation)
        x_left = ResidualBlock(width)(x_left)
        skips.append(x_left)
        x_left = layers.AveragePooling2D(2)(x_left)
        x_left = layers.Conv2D(width, 3, padding='same', dilation_rate=3)(x_left)  # Increased dilation

        # Encoder 2 (5x5 with increasing dilation)
        x_bottom = ResidualBlock(width)(x_bottom)
        skips.append(x_bottom)
        x_bottom = layers.AveragePooling2D(2)(x_bottom)
        x_bottom = layers.Conv2D(width, 5, padding='same', dilation_rate=3)(x_bottom)  # Increased dilation

    # ======================================#
    # Spatial Pyramid Pooling at Bottleneck #
    # ======================================#
    x_left_spp = SpatialPyramidPooling(x_left, levels=[1, 2, 4])
    x_bottom_spp = SpatialPyramidPooling(x_bottom, levels=[1, 2, 4])
    x = layers.Concatenate()([x_left_spp, x_bottom_spp])
    x = layers.Conv2D(widths[-1], 1, padding='same')(x)  # Reduce channels to match bottleneck width

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    # ===============#
    # Single Decoder #
    # ===============#
    for width in reversed(widths[:-1]):
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)

        skip = layers.Concatenate()([skips.pop(), skips.pop()])
        x = layers.Concatenate()([x, skip])
        x = ResidualBlock(width)(x)

    # Output noise prediction
    output = layers.Conv2D(3, 1, kernel_initializer='zeros')(x)

    return keras.Model([input_image, noise_variances], output, name="Loom-GenNet")

@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network

        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = keras.random.normal(
            shape=(num_images, image_size, image_size, 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        batch_size = tf.shape(images)[0]
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        batch_size = tf.shape(images)[0]
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_squared_error,
)
model.normalizer.adapt(full_dataset)

history = model.fit(
    full_dataset,
    epochs=num_epochs,
    callbacks=[
        # plot generated images at the end of each epoch
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
    ],
)
