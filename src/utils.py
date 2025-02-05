import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def load_data():
    try:
        data = pd.read_csv('data.csv')
        return data
    except FileNotFoundError:
        print('File not found')
        return None
    

def save_data(data):
    try:
        data.to_csv('data.csv', index=False)
        print('Data saved')
    except:
        print('Error saving data')


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def visualize_latent_space(encoder, data, labels):
    # Encode data
    encoded_data = encoder.predict(data)
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()


def save_model(model, name):
    try:
        model.save(name)
        print('Model saved')
    except:
        print('Error saving model')


def load_model(name):
    try:
        model = tf.keras.models.load_model(name)
        return model
    except:
        print('Error loading model')


