import sys
import os
import pickle
import numpy as np

with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "all_5_node_graphs.pickle"
        ),
        "rb",
) as f:
    all_5_node: np.ndarray = pickle.load(f)
with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "all_6_node_graphs.pickle"
        ),
        "rb",
) as f:
    all_6_node: np.ndarray = pickle.load(f)
del f
