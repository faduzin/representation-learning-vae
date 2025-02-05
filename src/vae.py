import tensorflow as tf

def build_encoder(input_shape, latent_dim):
    # Define encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(latent_dim)
    ])
    
    return encoder


def build_decoder(latent_dim, output_shape):
    # Define decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(tf.reduce_prod(output_shape), activation='sigmoid'),
        tf.keras.layers.Reshape(output_shape)
    ])
    
    return decoder


def build_vae(input_shape, latent_dim):
    # Define VAE
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)
    
    # Define sampling layer
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(mean))
            return mean + tf.exp(0.5 * log_var) * epsilon
        
    # Define VAE model
    inputs = tf.keras.layers.Input(shape=input_shape)
    mean, log_var = tf.split(encoder(inputs), num_or_size_splits=2, axis=1)
    z = Sampling()([mean, log_var])
    outputs = decoder(z)
    vae = tf.keras.Model(inputs, outputs)
    
    # Define loss
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, outputs), axis=(1, 2)))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    
    return vae


def build_beta_vae(input_shape, latent_dim, beta):
    # Define VAE
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)
    
    # Define sampling layer
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(mean))
            return mean + tf.exp(0.5 * log_var) * epsilon
        
    # Define VAE model
    inputs = tf.keras.layers.Input(shape=input_shape)
    mean, log_var = tf.split(encoder(inputs), num_or_size_splits=2, axis=1)
    z = Sampling()([mean, log_var])
    outputs = decoder(z)
    vae = tf.keras.Model(inputs, outputs)
    
    # Define loss
    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, outputs), axis=(1, 2)))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    vae_loss = reconstruction_loss + beta * kl_loss
    vae.add_loss(vae_loss)
    
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