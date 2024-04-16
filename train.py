# We will train the model here
import pandas as pd
import numpy as np
from utils import split_dataset

def main():
  train_set, test_set = split_dataset('data/data.csv')
  
if __name__ == '__main__':
  main()