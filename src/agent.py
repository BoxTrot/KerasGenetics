import sys
import warnings
import numpy as np
import keras
import keras.callbacks
import keras.layers
import keras.losses
import keras.metrics
import keras.utils

"""
The agent will contain the genome, the adjacency matrix, and exposes the interface for
an internal Keras Model.

-----------------------------------

Prototype data structure to hold the genetic material for each agent:

genes = [
    {
        "node_type": "Dense",
        "config": {},
        "modifiable_params": ["foo", "bar"],
    },
    {
        "node_type": "sequential_functional",
        "contents": [
            {
                "node_type": "Dense",
                "config": {},
                "modifiable_params": ["foo", "bar"],
            },
            {
                "node_type": "Dropout",
                "config": {},
                "modifiable_params": ["foo", "bar"],
            }
        ]
    }
]

Possible values for node_type:
    - Class name for any Keras layers
    - "sequential_functional", which means that each node_type listed in the "contents"
      attribute will be chained with the function API in index order 0->1->2->...->N
      
If the node_type is not "sequential_functional", the attributes "config" and
"modifiable_params" must exist.

If the node_type is "sequential_functional", the attribute "contents" must exist. 
"""


class Agent:
    def __init__(self):
        pass

    def gen_model(self):
        pass

    def gen_children(self):
        pass


def main(argv, *args):
    pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))
