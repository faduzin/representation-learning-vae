# Latent Space Exploration with Variational Autoencoders
This project explores representation learning using Variational Autoencoders (VAEs) on the Iris and Wine datasets. The focus is on understanding how VAEs compress high-dimensional tabular data into meaningful latent space representations. By experimenting with different VAE architectures, including Vanilla VAE and Beta-VAE, the project aims to visualize and analyze how well these models capture the underlying structure of the datasets.

## Table of Contents

1. [Background](#background)
2. [What is a Variational Autoencoder?](#what-is-a-variational-autoencoder?)
4. [Tools I Used](#tools-i-used)
5. [The Process](#the-process)
6. [The Analysis](#the-analysis)
7. [What I Learned](#what-i-learned)
8. [Skills Practiced](#skills-practiced)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Background

This project is designed to practice and apply knowledge of Variational Autoencoders (VAEs) in the context of representation learning. By working with well-known tabular datasets like Iris and Wine, the goal is to deepen the understanding of how VAEs learn to compress data into a meaningful latent space and how effectively these representations capture the underlying structure of the data.

## What is a Variational Autoencoder?

A Variational Autoencoder (VAE) is a type of generative model that learns to compress data into a lower-dimensional latent space and then reconstruct it back to its original form. Unlike traditional autoencoders, which learn fixed point representations, VAEs introduce a probabilistic approach by learning a distribution over the latent space. This allows VAEs not only to reconstruct existing data but also to generate new, similar data by sampling from the learned distribution.

### How does it works?

A VAE consists of three main components:

1. **Encoder:**  
   The encoder takes the input data (e.g., features from the Iris or Wine datasets) and maps it to a set of parameters—typically a **mean** (\(\mu\)) and **log-variance** (\(\log \sigma^2\))—that define a Gaussian distribution in the latent space. Instead of encoding data into a single point, the encoder defines a **probability distribution** for each input.

2. **Reparameterization Trick:**  
   To allow for backpropagation during training, VAEs use a technique called the **reparameterization trick**. This involves sampling a random variable \(\epsilon\) from a standard normal distribution and transforming it using the learned mean and variance:  
   \[
   z = \mu + \sigma \times \epsilon
   \]  
   This step ensures that the stochastic nature of sampling doesn't disrupt the gradient-based learning process.

3. **Decoder:**  
   The decoder takes the sampled latent vector \(z\) and tries to reconstruct the original input data. The quality of this reconstruction helps guide the learning process, ensuring the latent space captures meaningful features of the input.

### What Can VAEs Do?

1. **Representation Learning:**  
   VAEs learn compact, meaningful representations of data in the latent space. These representations can reveal underlying patterns and structures in the data, making them useful for tasks like clustering, visualization, and anomaly detection.

2. **Data Generation:**  
   Because VAEs learn a distribution over the latent space, they can generate **new, synthetic data** by sampling from this space. This makes VAEs powerful generative models capable of creating new examples that resemble the original dataset.

3. **Dimensionality Reduction:**  
   Similar to techniques like PCA (Principal Component Analysis), VAEs can reduce high-dimensional data to lower dimensions while preserving important information. However, VAEs provide a more flexible, non-linear approach to dimensionality reduction.

4. **Anomaly Detection:**  
   By measuring how well a VAE can reconstruct input data, it can identify anomalies—data points that don't fit well within the learned distribution typically result in higher reconstruction errors.

In this project, VAEs will be used primarily for **representation learning**, exploring how effectively they can compress the **Iris** and **Wine** datasets into meaningful latent spaces and how well these representations capture the datasets' intrinsic structures.

