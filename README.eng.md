# Latent Space Exploration with Variational Autoencoders
This project explores representation learning using Variational Autoencoders (VAEs) on the Iris and Wine datasets. The focus is on understanding how VAEs compress high-dimensional tabular data into meaningful latent space representations. By experimenting with VAE architectures, Beta-VAE, the project aims to visualize and analyze how well this model captures the underlying structure of the datasets.

## Table of Contents

1. [Background](#background)
2. [What is a Variational Autoencoder?](#what-is-a-variational-autoencoder?)
3. [Tools I Used](#tools-i-used)
4. [The Process](#the-process)
5. [The Analysis](#the-analysis)
6. [What I Learned](#what-i-learned)
7. [Skills Practiced](#skills-practiced)
8. [Conclusion](#conclusion)
9. [Contact](#contact)
10. [Repository Structure](#repository-structure)

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

## Tools I Used

- **Programming Language:** Python 3.12.9  
- **Interactive Environment:** Jupyter Notebook  
- **Libraries and Frameworks:**  
  - NumPy, Pandas for data manipulation  
  - Matplotlib for visualization  
  - scikit-learn for data preprocessing and metrics  
  - A deep learning library (e.g., TensorFlow/Keras or PyTorch) for implementing and training the VAE models  
- **Other Tools:** Git for version control

## The Process

1. **Data Loading & Preprocessing:**  
   The Iris and Wine datasets are loaded from the `data/` folder. Data preprocessing steps include normalization and splitting into training and test sets.

2. **Model Architecture:**  
   - **Encoder:** Transforms input features into latent space parameters (mean and log-variance).  
   - **Reparameterization:** Implements the trick to sample latent vectors while maintaining differentiability.  
   - **Decoder:** Reconstructs the original input from the latent vector.

3. **Training:**  
   The models are trained using a combined loss function that includes reconstruction loss (to ensure data fidelity) and KL divergence (to regularize the latent space). Hyperparameters such as learning rate, batch size, and the weight on the KL term (especially in Beta-VAE) are tuned in the notebooks.

4. **Evaluation:**  
   The trained models are evaluated by visualizing the latent space (e.g., scatter plots colored by class labels) and analyzing reconstruction quality. Comparisons between the Vanilla VAE and Beta-VAE architectures are made to assess the impact of regularization strength on learned representations.

## The Analysis

Visualizations stored in the `assets/` folder—such as loss curves, latent space plots, and reconstructed data comparisons—provide empirical evidence for the performance of the models. These images support the evaluation of how well the VAE models capture and reconstruct data from the Iris and Wine datasets.

## Detailed Analysis Conclusions

### Iris Analysis

- **Loss Behavior:**  
  The loss graph shows that the model learned effectively from the data.
<img src="assets\iris-assets\iris-loss.png" alt="Iris Loss" width="70%">

  
- **Latent Space Visualization:**  
  The latent space plot does not reveal well-defined clusters, even after the loss function stabilizes.
  <img src="assets\iris-assets\iris-latent-space.png" alt="Latent Space" width="100%">
  
- **Reconstruction Quality:**  
  The reconstructed data approximates the original data. It is observed that the blue class tends to cluster on the left, the green class centers, and the red class shifts to the right.
  <img src="assets\iris-assets\iris-reconstructed-pca.png" alt="Reconstructed PCA" width="70%">

### Wine Analysis

- **Encoding Patterns:**  
  The model successfully extracted encoding patterns as indicated by the stabilization of the loss.
<img src="assets\wine-assets\wine-loss.png" alt="Wine Loss" width="70%">
  
- **Reconstruction Characteristics:**  
  The reconstruction produced data that are highly centralized and tightly grouped.
<img src="assets\wine-assets\wine-reconstructed-pca.png" alt="Wine Reconstructed PCA" width="70%">
  
- **Latent Space Clustering:**  
  There is no evident formation of clusters in the latent space visualization.
<img src="assets\wine-assets\wine-latent-space.png" alt="Wine Latent Space" width="70%">

### General Creation Conclusions

- **Loss Value Consistency:**  
  The loss must always be a positive value; if not, there is an error in the model construction.
  
- **Impact of Data Scaling:**  
  The range in which the data are scaled directly affects the loss calculation. Initially, using standard scaling along with binary crossentropy resulted in loss values oscillating between positive and negative, leading the model to learn a tendency to centralize the reconstructed data (resulting in nearly identical output values). Switching to minmax scaling (0 to 1) resolved this issue, allowing the loss function to properly guide the learning process and producing reconstructed features that more closely match the original data.
  
- **Visualization Choices:**  
  Employing pairplots for datasets with many features is impractical. Therefore, only 5 arbitrarily chosen features were used in the visualizations, which proved sufficient to interpret the results.

## What I Learned

- **Representation Learning:**  
  I gained an in-depth understanding of how VAEs learn compressed representations of complex datasets.
  
- **Probabilistic Modeling:**  
  The project provided insights into the benefits and challenges of encoding data as distributions rather than fixed points.
  
- **Practical Implementation:**  
  Building, training, and evaluating deep learning models using Jupyter notebooks and modular code enhanced my practical skills.
  
- **Visualization and Analysis:**  
  The experience improved my ability to visualize high-dimensional data and interpret latent space structures effectively.

## Skills Practiced

- Data Preprocessing and Exploration  
- Deep Learning Model Design and Implementation  
- Experimentation with Generative Models (Vanilla VAE vs. Beta-VAE)  
- Use of Python and Jupyter Notebooks for prototyping and analysis  
- Effective Data Visualization Techniques

## Conclusion

This project demonstrates the potential of Variational Autoencoders for effective representation learning on classical tabular datasets. The empirical evidence (available in the `assets/` folder) confirms that while the models can learn meaningful representations and reconstruct data well, challenges such as latent space clustering and proper scaling must be addressed. Future work may include exploring additional datasets, refining model architectures, or applying the learned representations to downstream tasks like clustering and anomaly detection.


## Contact

If you have any questions or feedback, feel free to reach out:\
[GitHub](https://github.com/faduzin) | [LinkedIn](https://www.linkedin.com/in/ericfadul/) | [eric.fadul@gmail.com](mailto\:eric.fadul@gmail.com)

## Repository Structure

- **assets/**: Contains images, figures, and visualizations generated during analysis.
- **data/**: Includes the Iris and Wine datasets used in the project.
- **notebooks/**: Jupyter notebooks documenting experiments, model training, and analysis.
- **src/**: Source code for model definitions, training routines, and utility functions.
- **.gitignore**: Specifies files and directories that Git should ignore.
- **LICENSE**: The MIT license under which this project is released.
- **README.md**: This file, providing an overview of the project and instructions for use.