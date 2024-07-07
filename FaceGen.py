# Import libraries and modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the CelebA dataset (assuming you have downloaded it)
celeba_dir = 'list_attr_celeba.csv'
img_dir = celeba_dir + '/img_align_celeba'

# Load the images
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
x_train = []
for file in img_files:
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    x_train.append(img)
x_train = np.array(x_train) / 255.0

# Load and preprocess the face dataset
(x_train, _), (_, _) = keras.datasets.celeba.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train[:, 20:-20, 20:-20]  # crop the images
x_train = tf.image.resize(x_train, (64, 64))  # resize the images

# Define the generator model
def make_generator_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((8, 8, 256)))
    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Define the discriminator model
def make_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    return model

# Create the generator and the discriminator
generator = make_generator_model()
discriminator = make_discriminator_model()

# Define the loss function and the optimizer
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

# Train the GAN
BATCH_SIZE = 128
BUFFER_SIZE = 60000

# Create a tf.data.Dataset object to load the images and shuffle them
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

EPOCHS = 100

for epoch in range(EPOCHS):
    for images in train_dataset:
        gen_loss, disc_loss = train_step(images)

    if epoch % 10 == 0:
        print('Epoch:', epoch)
        noise = tf.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        plt.imshow(generated_image[0].numpy())
        plt.show()