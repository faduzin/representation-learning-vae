import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats import norm
from tensorflow.keras import (
    layers, 
    models,
    metrics, 
    optimizers, 
    losses, 
    callbacks
)
from sklearn.decomposition import PCA

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
def build_encoder(latent_dim, input_shape, topology=[256, 128, 64]):
    encoder_inputs = layers.Input(
        shape=(input_shape,), name="encoder_input_layer"
    )

    x = layers.Dense(topology[0], activation='relu')(encoder_inputs)
    x = layers.Dense(topology[1], activation='relu')(x)
    x = layers.Dense(topology[2], activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean_layer")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var_layer")(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    
    return encoder


def build_decoder(latent_dim, input_shape, topology=[256, 128, 64]):
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input_layer")

    x = layers.Dense(topology[2], activation='relu')(decoder_inputs)
    x = layers.Dense(topology[1], activation='relu')(x)
    x = layers.Dense(topology[0], activation='relu')(x)

    decoder_outputs = layers.Dense(input_shape, activation='sigmoid')(x)

    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    return decoder

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return z_mean, z_log_var, reconstructed
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            beta = 1
            reconstruction_loss = tf.reduce_mean(
                beta * losses.binary_crossentropy(data, reconstruction, axis=1)
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return { m.name: m.result() for m in self.metrics }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        beta = 1
        reconstruction_loss = tf.reduce_mean(
            beta * losses.binary_crossentropy(data, reconstruction, axis=1)
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1
            )
        )

        total_loss = reconstruction_loss + kl_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }


def model_checkpoint_callback(filepath):
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
        monitor="total_loss",
        mode="min",
        verbose=0
    )
    print("Model checkpoint callback created.")
    return model_checkpoint


def predict(vae, data, label, n_to_predict=1000):
    n_to_predict = 5000
    example = data[:n_to_predict]
    example_labels = label[:n_to_predict]
    z_mean, z_log_var, reconstructions = vae.predict(example)
    return z_mean, z_log_var, reconstructions, example_labels


def plot_latent_space(encoder, example, example_labels):
    z_mean, z_var, z = encoder.predict(example)
    p = norm.cdf(z)

    figsize = 8
    fig = plt.figure(figsize=(figsize * 2, figsize))
    ax = fig.add_subplot(1, 2, 1)
    plot_1 = ax.scatter(
        z[:, 0], z[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=15
    )
    plt.colorbar(plot_1)
    ax = fig.add_subplot(1, 2, 2)
    plot_2 = ax.scatter(
        p[:, 0], p[:, 1], cmap="rainbow", c=example_labels, alpha=0.8, s=15
    )
    plt.show()


def plot_reduced_pca(data, labels):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Scatter plot of the PCA projection
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Class Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Encoded Data")
    plt.show()

    # Print explained variance ratio
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")