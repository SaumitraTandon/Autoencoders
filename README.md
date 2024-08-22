# AutoEncoders

This project demonstrates the implementation of a Stacked AutoEncoder (SAE) using PyTorch, applied to the MovieLens ML-100K dataset. The goal is to compress and reconstruct user rating data, potentially for recommendation systems.

## Dataset

The dataset used is the [MovieLens 100K](http://files.grouplens.org/datasets/movielens/ml-100k.zip), a classic dataset in the recommendation systems community.

### Downloading the Dataset

The dataset is automatically downloaded and unzipped during the execution of the notebook using the following commands:

```bash
!wget "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
!unzip ml-100k.zip
```

The dataset includes user ratings for various movies, stored in several files.

## Model: Stacked AutoEncoder (SAE)

The SAE is trained to learn compressed representations of user ratings. This model consists of multiple layers designed to reduce the dimensionality of the input data and then reconstruct it.

### Model Training

The training process minimizes the reconstruction error using Mean Squared Error (MSE) as the loss function. The model is trained on the training set while evaluating the performance on a test set.

### Testing the SAE

After training, the model's performance is evaluated on a separate test set using Root Mean Squared Error (RMSE) to measure the difference between the predicted and actual ratings.

The test evaluation script outputs the test loss, which is an indicator of how well the model generalizes to unseen data.

## Requirements

- Python 3.x
- PyTorch
- NumPy

You can install the required packages using pip:

```bash
pip install torch numpy
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Execute the Jupyter Notebook `AutoEncoders.ipynb`.

```bash
jupyter notebook AutoEncoders.ipynb
```

3. Follow the steps in the notebook to download the dataset, train the model, and evaluate its performance.

## Results

The final output of the notebook will display the test loss after evaluating the model on the test set. A lower test loss indicates better performance.

## Contribution

Feel free to fork the repository, create pull requests, or submit issues. Contributions to improve the model or add features are welcome.
