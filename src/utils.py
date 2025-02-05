import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print('Data loaded successfully.')
        return data
    except FileNotFoundError:
        print('File not found.')
        return None
    

def save_data(data, file_path, file_name):
    try:
        file = file_path + file_name
        data.to_csv('data.csv', index=False)
        print('Data saved successfully.')
    except:
        print('Error saving data.')


def data_info(data):
    print(data.info())
    print("-" * 50)
    print(data.describe())
    print("-" * 50)
    print(data.head(5))
    print("-" * 50)
    print("Data shape: ",data.shape)
    print("Amount of duplicates: ", data.duplicated().sum())


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
        print('Model saved successfully.')
    except:
        print('Error saving model.')


def load_model(name):
    try:
        model = tf.keras.models.load_model(name)
        print('Model loaded successfully.')
        return model
    except:
        print('Error loading model.')


