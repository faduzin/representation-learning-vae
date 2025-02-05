import tensorflow as tf


def evaluate_reconstruction(vae, data):
    # Evaluate reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, vae(data)), axis=(1, 2)))
    return reconstruction_loss


def evaluate_kl_divergence(vae, data):
    # Evaluate KL divergence
    mean, log_var = tf.split(vae.encoder(data), num_or_size_splits=2, axis=1)
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return kl_loss


def compare_models(vae1, vae2, data):
    # Compare reconstruction loss of two models
    loss1 = evaluate_reconstruction(vae1, data)
    loss2 = evaluate_reconstruction(vae2, data)
    return loss1, loss2


def generate_samples(decoder, num_samples=1):
    # Generate samples from the decoder
    latent_dim = decoder.input_shape[1]
    latent_samples = tf.random.normal(shape=(num_samples, latent_dim))
    samples = decoder(latent_samples)
    return samples