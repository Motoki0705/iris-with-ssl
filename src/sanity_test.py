import torch
import numpy as np
import pandas as pd
from sklearn import datasets

print("Torch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

iris = datasets.load_iris()
print("Iris dataset shape:", iris.data.shape)
