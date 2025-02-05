import tensorflow as tf

# Define sampling layer
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

def build_encoder(input_shape, latent_dim):
    # Define encoder
    try:
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(latent_dim)
        ])
        print('Encoder built successfully.')
    except:
        print('Error building encoder')
        return None
    
    return encoder


def build_decoder(latent_dim, output_shape):
    # Define decoder
    try:
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_shape[0], activation='softmax'),
        ])
        print('Decoder built successfully.')
    except:
        print('Error building decoder.')
        return None
    
    return decoder


def build_vae(input_shape, latent_dim, beta=1.0):
    # Define VAE
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)

    if encoder is None or decoder is None:
        return None
    
    # Define VAE model
    try:
        inputs = tf.keras.layers.Input(shape=input_shape)
        encoder_output = encoder(inputs)
        mean_log_var = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(encoder_output)
        mean = mean_log_var[0]
        log_var = mean_log_var[1]
        z = Sampling()([mean, log_var])
        outputs = decoder(z)
        vae = tf.keras.Model(inputs, outputs)
        print('VAE built successfully.')
    except Exception as e:
        print(f'Error building VAE: {e}')
        return None
    
    # Define loss
    try:
        # Reconstruction loss (use vae.input instead of `inputs`)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(vae.input, vae.output)
        )

        # KL Divergence loss
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))

        # Final loss
        vae_loss = reconstruction_loss + beta * kl_loss

        # Add loss before compiling
        vae.add_loss(vae_loss)

        # Compile VAE
        vae.compile(optimizer=tf.keras.optimizers.Adam())
        print('Loss defined successfully.')
    except:
        print('Error defining loss.')
        return None
    
    return vae


def train_vae(vae, 
              data, 
              epochs=100,
              optimizer='adam', 
              batch_size=32):
    
    match optimizer:
        case 'adam':
            pass
        case 'rmsprop':
            pass
        case _:
            print('Invalid optimizer')
            return None
    
    vae.compile(optimizer=optimizer)
    history = vae.fit(data, epochs=epochs, batch_size=batch_size)
    
    return history