import pandas as pd
import numpy as np

def load_dataset(path):
  """
  Load the dataset from the given path.
  Args:
    param path: the path to the dataset
  Returns:
    numpy Array: the dataset
  """

  dataset = pd.read_csv(path)
  return dataset.to_numpy()

def split_dataset(path, split_ratio=0.8):
  """
  Split the dataset into training and testing dataset.
  Args:
    param path: the path to the dataset
    param split_ratio: the ratio to split the dataset(between 0.1 and 0.9)
  Returns:
    tuple: the training and testing dataset
  """

  if split_ratio < 0.1 or split_ratio > 0.9:
    raise ValueError("split_ratio must be between 0.1 and 0.9")

  dataset = load_dataset(path)
  num_train = int(len(dataset) * split_ratio)
  train_set = dataset[:num_train]
  test_set = dataset[num_train:]
  return train_set, test_set