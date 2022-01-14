# Data wrangling
import pandas as pd

# Numerical operations
import numpy as np

# Random selections
import random

# Quick value count calculator
from collections import Counter

# Tree growth tracking
from tqdm import tqdm

# Accuracy metrics
from sklearn.metrics import precision_score, recall_score


from main import RandomForestTree

tree = RandomForestTree(
                X=D,
                Y=Y,
                min_samples_split=5,
                max_depth=100,
                X_features_fraction=0.5
            )


tree.grow_tree()

tree.X
tree.Y
Counter(tree.Y)
min_samples_split=5,
max_depth=2,
n_trees=1,
X_features_fraction=0.5


df = tree.X.copy()

# Default best feature and split

best_feature = None
best_value = None
best_linear = None

# Getting a random subsample of features
df = tree.X

n_features = len(df.columns)
n_ft = int(tree.n_features * tree.X_features_fraction)

# Selecting random features without repetition
features_subsample = random.sample(tree.features, n_ft)
best_feature = features_subsample


if best_feature is not None:
    A = df.loc[best_feature, best_feature]
    D = np.diag(A.sum(axis = 0))
    U, V, D = np.linalg.svd(D-A)
    best_linear = D[:,-2]

    tree.best_linear = best_linear

    best_vec = df.loc[:, best_feature].dot(best_linear.reshape(-1,1))

    best_value = tree.split_val(best_vec)



if (tree.depth < tree.max_depth) and (tree.n >= tree.min_samples_split):

    # Getting the best split
    best_feature, best_value = tree.spec_split()

    if best_feature is not None:
        # Saving the best split to the current node
        tree.best_feature = best_feature
        tree.best_value = best_value
        vec = tree.X.loc[:, best_feature].dot(tree.best_linear.reshape(-1, 1))[0]

        # Getting the left and right dataframe indexes
        # left_index, right_index = tree.X.loc[vec <= best_value, vec <= best_value].index,\
        #                          tree.X.loc[vec > best_value, vec > best_value].index

        # Extracting the left X and right X
        left_X, right_X = tree.X.loc[vec <= best_value, vec <= best_value], \
                          tree.X.loc[vec > best_value, vec > best_value]

        # Reseting the indexes
        # left_X.reset_index(inplace=True, drop=True)
        # right_X.reset_index(inplace=True, drop=True)

        # Extracting the left Y and the right Y
        left_Y, right_Y = tree.Y.loc[vec <= best_value,], tree.Y.loc[vec > best_value,]

        # Creating the left and right nodes
        left = RandomForestTree(
            left_X,
            left_Y,
            depth=0,
            max_depth=tree.max_depth,
            min_samples_split=tree.min_samples_split,
            node_type='left_node',
            rule=None
        )

        tree.left = left
        tree.left.grow_tree()

        right = RandomForestTree(
            right_X,
            right_Y,
            depth=0,
            max_depth=tree.max_depth,
            min_samples_split=tree.min_samples_split,
            node_type='right_node',
            rule=None
        )

        tree.right = right
        tree.right.grow_tree()