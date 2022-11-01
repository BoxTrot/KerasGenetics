import sys
from copy import deepcopy

import keras

import genome

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
    def __init__(self, genes: genome.Genome, deep_copy=False):
        if deep_copy:
            self.genes = deepcopy(genes)
        else:
            self.genes = genes
        self.model = keras.Model()

    def gen_model(self):
        self.model = self.genes.gen_model()

    def gen_children(
        self,
        clones: int = 2,
        variants: int = 2,
        graph_mutations: int = 1,
        num_mut_p=0.5,
        cat_mut_p=0.05,
        mut_power=0.01,
        rng=None,
    ):
        child_genomes = self.genes.gen_children(
            clones=clones,
            variants=variants,
            graph_mutations=graph_mutations,
            num_mut_p=num_mut_p,
            cat_mut_p=cat_mut_p,
            mut_power=mut_power,
            rng=rng,
        )
        return [Agent(g) for g in child_genomes]


def main(argv, *args):
    pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))
