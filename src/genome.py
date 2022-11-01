import sys
import warnings
from copy import deepcopy

import numpy as np
from scipy.stats import rankdata
from tensorflow import keras

import topo
import utils

"""
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
_node_types = {
    "Activation": {"class": keras.layers.Activation, "default": ("relu",)},
    "ActivityRegularization": {
        "class": keras.layers.ActivityRegularization,
        "default": None,
    },
    "AdditiveAttention": {"class": keras.layers.AdditiveAttention, "default": None},
    "AlphaDropout": {"class": keras.layers.AlphaDropout, "default": (0.5,)},
    "Attention": {"class": keras.layers.Attention, "default": None},
    "Dense": {"class": keras.layers.Dense, "default": (1,)},
    "Dropout": {"class": keras.layers.Dropout, "default": (0.5,)},
    "ELU": {"class": keras.layers.ELU, "default": None},
    "GRU": {"class": keras.layers.GRU, "default": (1,)},
    "GRUCell": {"class": keras.layers.GRUCell, "default": (1,)},
    "GaussianDropout": {"class": keras.layers.GaussianDropout, "default": (0.5,)},
    "GaussianNoise": {"class": keras.layers.GaussianNoise, "default": (1,)},
    "LayerNormalization": {"class": keras.layers.LayerNormalization, "default": None},
    "PReLU": {"class": keras.layers.PReLU, "default": None},
    "ReLU": {"class": keras.layers.ReLU, "default": None},
    "Softmax": {"class": keras.layers.Softmax, "default": None},
    "ThresholdedReLU": {"class": keras.layers.ThresholdedReLU, "default": None},
}
_valid_node_types = _node_types.keys()
_default_rng = np.random.default_rng()


def warn_raise(message: str, warnlevel: int):
    """
    Simplifies choosing a warning level.
    :param message: What message to give to the warning or exception, if one is raised.
    :param warnlevel: What type of error/warning message to raise, if at all.
        0 = silent, 1 = warnings, 2 or any other value = exceptions
    :return: Nothing.for
    """
    if warnlevel == 1:
        warnings.warn(message)
    elif warnlevel == 0:
        return
    else:
        raise UserWarning(message)


class GenomeMaster:
    # TODO: Change to use content from tf.keras.utils.serialize_keras_object
    def __init__(self, rng: np.random.Generator = None):
        if rng is None:
            global _default_rng
            self.rng = _default_rng
        else:
            self.rng = rng
        self.master = {
            "Dense": {
                "units": {
                    "type": "int",
                    "range": (0, 200),
                },
                "activation": {
                    "type": "categorical",
                    "values": (
                        "elu",
                        "exponential",
                        "gelu",
                        "linear",
                        "relu",
                        "selu",
                        "sigmoid",
                        "softplus",
                        "swish",
                    ),
                    "probability": tuple([1 / 9] * 9),
                },
                "kernel_initializer": {
                    "type": "categorical",
                    "values": (
                        {"class_name": "GlorotNormal", "config": {"seed": None}},
                        {"class_name": "GlorotUniform", "config": {"seed": None}},
                        {"class_name": "HeNormal", "config": {"seed": None}},
                        {"class_name": "HeUniform", "config": {"seed": None}},
                        {"class_name": "LecunNormal", "config": {"seed": None}},
                        {"class_name": "LecunUniform", "config": {"seed": None}},
                        {
                            "class_name": "VarianceScaling",
                            "config": {
                                "scale": 1.0,
                                "mode": "fan_in",
                                "distribution": "truncated_normal",
                                "seed": None,
                            },
                        },
                    ),
                    "probability": tuple([1 / 7] * 7),
                },
                "bias_initializer": {
                    "type": "categorical",
                    "values": (
                        {"class_name": "Zeros", "config": {}},
                        {"class_name": "GlorotNormal", "config": {"seed": None}},
                        {"class_name": "GlorotUniform", "config": {"seed": None}},
                        {"class_name": "HeNormal", "config": {"seed": None}},
                        {"class_name": "HeUniform", "config": {"seed": None}},
                        {"class_name": "LecunNormal", "config": {"seed": None}},
                        {"class_name": "LecunUniform", "config": {"seed": None}},
                        {
                            "class_name": "VarianceScaling",
                            "config": {
                                "scale": 1.0,
                                "mode": "fan_in",
                                "distribution": "truncated_normal",
                                "seed": None,
                            },
                        },
                    ),
                    "probability": tuple([0.6] + [0.4 / 7] * 7),
                },
            },
            "Dropout": {
                "rate": {
                    "type": "float",
                    "range": (0.0, 0.7),
                },
            },
            "AlphaDropout": {
                "rate": {
                    "type": "float",
                    "range": (0.0, 0.7),
                },
            },
        }

    def gen_value(self, node_type: str, attr: str):
        if self.master[node_type][attr]["type"] == "categorical":
            return self.master[node_type][attr]["values"][
                self.rng.choice(
                    len(self.master[node_type][attr]["values"]),
                    p=self.master[node_type][attr]["probability"],
                )
            ]
        elif self.master[node_type][attr]["type"] == "int":
            return self.rng.integers(*self.master[node_type][attr]["range"])
        elif self.master[node_type][attr]["type"] == "float":
            return self.rng.uniform(*self.master[node_type][attr]["range"])
        else:
            raise UserWarning(
                "value type {} of attribute {} in node type {} is unrecognised!".format(
                    self.master[node_type][attr]["type"], attr, node_type
                )
            )

    def mutate_value(self, node_type: str, attr: str, value, power=0.01):
        """
        Picks a new value for the attribute from a normal distribution centered on the
        old value with the standard deviation power*range
        :param node_type: The node type, e.g. "Dense".
        :param attr: The attribute.
        :param value: The old value, the mean of the normal distribution.
        :param power: Standard deviation of the normal distribution, expressed as a
            fraction of the attribute's range
        :return: The new value.
        """
        if self.master[node_type][attr]["type"] == "categorical":
            raise UserWarning(
                "Cannot generate value for categorical attribute {} in {}".format(attr, node_type)
            )
        else:
            _range = self.master[node_type][attr]["range"]
            _val = max(min(_range[1], value), _range[0])
            _out = self.rng.normal(_val, abs(_range[1] - _range[0]) * power)
            if self.master[node_type][attr]["type"] == "int":
                return max(min(_range[1], round(_out)), _range[0])
            else:
                return max(min(_range[1], _out), _range[0])

    def check_value(self, node_type: str, attr: str, value) -> bool:
        if node_type not in _valid_node_types:
            return False
        elif attr not in self.master[node_type].keys():
            return False
        elif (
            self.master[node_type][attr]["type"] == "int"
            or self.master[node_type][attr]["type"] == "float"
        ):
            _range = self.master[node_type][attr]["range"]
            if _range[0] <= value <= _range[1]:
                return True
            else:
                return False
        elif self.master[node_type][attr]["type"] == "categorical":
            if value in self.master[node_type][attr]["values"]:
                return True
            else:
                return False
        else:
            return False

    def is_cat(self, node_type: str, attr: str) -> bool:
        if self.master[node_type][attr]["type"] == "categorical":
            return True
        else:
            return False

    def is_num(self, node_type: str, attr: str) -> bool:
        if (
            self.master[node_type][attr]["type"] == "int"
            or self.master[node_type][attr]["type"] == "float"
        ):
            return True
        else:
            return False


class Genome:
    __default_genome_master = GenomeMaster()

    def __init__(
        self,
        input_layer: keras.layers.InputLayer,
        output_layer: keras.layers.Layer,
        genes: list[dict] = None,
        adj_matrix: np.ndarray = None,
        adj_mat_check: bool = True,
        master: GenomeMaster = None,
    ):
        """

        :param input_layer: The input layer to use.
        :param output_layer: The input layer to use.
        :param genes:
        :param adj_matrix: The adjacency matrix for the graph representing the model.
            Must be square and the same size as the number of genes + 2. The input node
            is assumed to be at index 0, and the output node is assumed to be at index
            -1.
        :param adj_mat_check: Whether to properly check the validity of the passed
            adjacency matrix with topo.is_valid_adj_matrix(). Otherwise, just checks the
            shape of the adjacency matrix.
        """
        if master is None:
            self.master = Genome.__default_genome_master
        else:
            self.master = master
        if genes is None:
            # TODO: Change to use content from tf.keras.utils.serialize_keras_object
            if adj_matrix is None or adj_matrix.shape[0] == 5:
                self._genes = [
                    {
                        "node_type": "sequential_functional",
                        "contents": [
                            {
                                "node_type": "Dense",
                                "config": {
                                    "name": "dense_0",
                                    "trainable": True,
                                    "dtype": "float32",
                                    "units": 128,
                                    "activation": "swish",
                                    "use_bias": True,
                                    "kernel_initializer": {
                                        "class_name": "GlorotUniform",
                                        "config": {"seed": None},
                                    },
                                    "bias_initializer": {
                                        "class_name": "Zeros",
                                        "config": {},
                                    },
                                    "kernel_regularizer": None,
                                    "bias_regularizer": None,
                                    "activity_regularizer": None,
                                    "kernel_constraint": None,
                                    "bias_constraint": None,
                                },
                                "modifiable_params": [
                                    "units",
                                    "activation",
                                    "kernel_initializer",
                                ],
                            },
                            {
                                "node_type": "Dropout",
                                "config": {
                                    "name": "dropout_0",
                                    "trainable": True,
                                    "dtype": "float32",
                                    "rate": 0.5,
                                    "noise_shape": None,
                                    "seed": None,
                                },
                                "modifiable_params": ["rate"],
                            },
                        ],
                    },
                    {
                        "node_type": "sequential_functional",
                        "contents": [
                            {
                                "node_type": "Dense",
                                "config": {
                                    "name": "dense_1",
                                    "trainable": True,
                                    "dtype": "float32",
                                    "units": 64,
                                    "activation": "swish",
                                    "use_bias": True,
                                    "kernel_initializer": {
                                        "class_name": "GlorotUniform",
                                        "config": {"seed": None},
                                    },
                                    "bias_initializer": {
                                        "class_name": "Zeros",
                                        "config": {},
                                    },
                                    "kernel_regularizer": None,
                                    "bias_regularizer": None,
                                    "activity_regularizer": None,
                                    "kernel_constraint": None,
                                    "bias_constraint": None,
                                },
                                "modifiable_params": [
                                    "units",
                                    "activation",
                                    "kernel_initializer",
                                ],
                            },
                            {
                                "node_type": "Dropout",
                                "config": {
                                    "name": "dropout_1",
                                    "trainable": True,
                                    "dtype": "float32",
                                    "rate": 0.5,
                                    "noise_shape": None,
                                    "seed": None,
                                },
                                "modifiable_params": ["rate"],
                            },
                        ],
                    },
                    {
                        "node_type": "sequential_functional",
                        "contents": [
                            {
                                "node_type": "Dense",
                                "config": {
                                    "name": "dense_2",
                                    "trainable": True,
                                    "dtype": "float32",
                                    "units": 32,
                                    "activation": "swish",
                                    "use_bias": True,
                                    "kernel_initializer": {
                                        "class_name": "GlorotUniform",
                                        "config": {"seed": None},
                                    },
                                    "bias_initializer": {
                                        "class_name": "Zeros",
                                        "config": {},
                                    },
                                    "kernel_regularizer": None,
                                    "bias_regularizer": None,
                                    "activity_regularizer": None,
                                    "kernel_constraint": None,
                                    "bias_constraint": None,
                                },
                                "modifiable_params": [
                                    "units",
                                    "activation",
                                    "kernel_initializer",
                                ],
                            },
                            {
                                "node_type": "Dropout",
                                "config": {
                                    "name": "dropout_2",
                                    "trainable": True,
                                    "dtype": "float32",
                                    "rate": 0.5,
                                    "noise_shape": None,
                                    "seed": None,
                                },
                                "modifiable_params": ["rate"],
                            },
                        ],
                    },
                ]
            elif adj_matrix.shape[0] > 3:
                if adj_matrix.shape[0] < 8:
                    self._genes = []
                    dims = [16, 32, 64, 96, 128, 160]
                    for i in range(adj_matrix.shape[0] - 2):
                        self._genes.insert(
                            0,
                            {
                                "node_type": "sequential_functional",
                                "contents": [
                                    {
                                        "node_type": "Dense",
                                        "config": {
                                            "name": "dense_{}".format(adj_matrix.shape[0] - 3 - i),
                                            "trainable": True,
                                            "dtype": "float32",
                                            "units": dims[i],
                                            "activation": "swish",
                                            "use_bias": True,
                                            "kernel_initializer": {
                                                "class_name": "GlorotUniform",
                                                "config": {"seed": None},
                                            },
                                            "bias_initializer": {
                                                "class_name": "Zeros",
                                                "config": {},
                                            },
                                            "kernel_regularizer": None,
                                            "bias_regularizer": None,
                                            "activity_regularizer": None,
                                            "kernel_constraint": None,
                                            "bias_constraint": None,
                                        },
                                        "modifiable_params": [
                                            "units",
                                            "activation",
                                            "kernel_initializer",
                                        ],
                                    },
                                    {
                                        "node_type": "Dropout",
                                        "config": {
                                            "name": "dropout_{}".format(
                                                adj_matrix.shape[0] - 3 - i
                                            ),
                                            "trainable": True,
                                            "dtype": "float32",
                                            "rate": 0.5,
                                            "noise_shape": None,
                                            "seed": None,
                                        },
                                        "modifiable_params": ["rate"],
                                    },
                                ],
                            },
                        )
                else:
                    self._genes = []
                    for i in range(adj_matrix.shape[0] - 2):
                        self._genes.insert(
                            0,
                            {
                                "node_type": "sequential_functional",
                                "contents": [
                                    {
                                        "node_type": "Dense",
                                        "config": {
                                            "name": "dense_{}".format(adj_matrix.shape[0] - 3 - i),
                                            "trainable": True,
                                            "dtype": "float32",
                                            "units": 32,
                                            "activation": "swish",
                                            "use_bias": True,
                                            "kernel_initializer": {
                                                "class_name": "GlorotUniform",
                                                "config": {"seed": None},
                                            },
                                            "bias_initializer": {
                                                "class_name": "Zeros",
                                                "config": {},
                                            },
                                            "kernel_regularizer": None,
                                            "bias_regularizer": None,
                                            "activity_regularizer": None,
                                            "kernel_constraint": None,
                                            "bias_constraint": None,
                                        },
                                        "modifiable_params": [
                                            "units",
                                            "activation",
                                            "kernel_initializer",
                                        ],
                                    },
                                    {
                                        "node_type": "Dropout",
                                        "config": {
                                            "name": "dropout_{}".format(
                                                adj_matrix.shape[0] - 3 - i
                                            ),
                                            "trainable": True,
                                            "dtype": "float32",
                                            "rate": 0.5,
                                            "noise_shape": None,
                                            "seed": None,
                                        },
                                        "modifiable_params": ["rate"],
                                    },
                                ],
                            },
                        )
            else:
                raise UserWarning("Adj matrix must be larger than 3!")
        else:
            if self.check_gene_integrity(genes, 1) and (
                (adj_matrix is None) or ((len(genes) + 2) == adj_matrix.shape[0])
            ):
                self._genes = genes
            else:
                raise UserWarning("input argument genes was invalid! contents: {}".format(genes))
        self.input_layer = deepcopy(input_layer)
        self.output_layer = deepcopy(output_layer)
        if len(self._genes) + 2 == 5:
            self.precomp: np.ndarray = utils.all_5_node
        elif len(self._genes) + 2 == 6:
            self.precomp: np.ndarray = utils.all_6_node
        else:
            self.precomp = None
        if adj_matrix is None:
            # if self.precomp is None:
            # Generates an adjacency matrix representing a straight graph with path
            # 0->1, 1->2, 2->3, etc.
            self.adj_matrix = np.eye(len(self._genes) + 2, k=1, dtype="B")
            # else:
            #     self.adj_matrix = self.precomp[
            #         _default_rng.integers(low=0, high=self.precomp.shape[0])
            #     ]
        elif adj_mat_check:
            if topo.is_valid_adj_matrix(adj_matrix, warning_lvl=2):
                self.adj_matrix = adj_matrix
            else:
                raise UserWarning("Invalid matrix passed!")
        else:
            if (
                (len(adj_matrix.shape) != 2)
                or (adj_matrix.shape[0] != adj_matrix.shape[1])
                or (adj_matrix.shape[0] != (len(self._genes) + 2))
            ):
                raise UserWarning("Invalid matrix passed!")
            else:
                self.adj_matrix = adj_matrix

    @staticmethod
    def check_gene_integrity(genes: list[dict], warnlevel: int = 1) -> bool:
        """

        :param genes:
        :param warnlevel: If any issues are found in the input, how to raise them if at all.
            0 = silent, 1 = warnings, 2 or any other value = exceptions
        :return: True if integrity good, otherwise False
        """
        global _valid_node_types
        for entry in genes:
            entry_keys = entry.keys()
            if "node_type" in entry_keys:
                if entry["node_type"] == "sequential_functional":
                    if "contents" in entry_keys:
                        if isinstance(entry["contents"], list):
                            if not Genome.check_gene_integrity(entry["contents"], warnlevel):
                                warn_raise(
                                    'invalid entries in "content" in {}'.format(entry),
                                    warnlevel,
                                )
                                return False
                        else:
                            warn_raise(
                                '"contents" in {} is not a list!'.format(entry),
                                warnlevel,
                            )
                            return False
                    else:
                        warn_raise('"contents" not found in {}'.format(entry), warnlevel)
                        return False
                elif entry["node_type"] in _valid_node_types:
                    if "config" not in entry_keys:
                        warn_raise('"config" not found in {}'.format(entry), warnlevel)
                        return False
                    elif not isinstance(entry["config"], dict):
                        warn_raise('"config" in {} is not a dict!'.format(entry), warnlevel)
                        return False
                    if "modifiable_params" not in entry_keys:
                        warn_raise(
                            '"modifiable_params" not found in {}'.format(entry),
                            warnlevel,
                        )
                        return False
                    elif not isinstance(entry["modifiable_params"], list):
                        warn_raise(
                            '"modifiable_params" in {} is not a list!'.format(entry),
                            warnlevel,
                        )
                        return False
                else:
                    warn_raise(
                        "{} is not a valid node type, or has not yet been implemented!".format(
                            entry["node_type"]
                        ),
                        warnlevel,
                    )
                    return False
            else:
                warn_raise('"node_type" not found in {}'.format(entry), warnlevel)
                return False
        return True

    @staticmethod
    def __layers_from_genes(genes: list[dict]) -> list:
        global _valid_node_types, _node_types
        output = []
        for entry in genes:
            if entry["node_type"] == "sequential_functional":
                output.append(Genome.__layers_from_genes(entry["contents"]))
            elif entry["node_type"] in _valid_node_types:
                if _node_types[entry["node_type"]]["default"] is None:
                    output.append(
                        _node_types[entry["node_type"]]["class"]().from_config(entry["config"])
                    )
                else:
                    output.append(
                        _node_types[entry["node_type"]]["class"](
                            *_node_types[entry["node_type"]]["default"]
                        ).from_config(entry["config"])
                    )
        return output

    def layers_from_genes(self) -> list:
        return Genome.__layers_from_genes(self._genes)

    def gen_model(self) -> keras.Model:
        return topo.model_from_graph_and_layers(
            adj_matrix=self.adj_matrix,
            in_node=0,
            out_node=self.adj_matrix.shape[0] - 1,
            input_layer=self.input_layer,
            layer_list=self.layers_from_genes(),
            output_layer=self.output_layer,
        )

    @staticmethod
    def get_order(adj_matrix: np.ndarray) -> np.ndarray:
        return rankdata(topo.average_pos(adj_matrix, exclude_in_out=True), method="ordinal") - 1

    def get_reordered(self, new_order: np.ndarray) -> list:
        geneordering = np.argsort(
            topo.average_pos(self.adj_matrix, exclude_in_out=True), kind="stable"
        )
        if len(new_order) == len(self._genes):
            ordered_genes = [
                self._genes[i] for i in geneordering
            ]  # genes now in order of closeness to origin in graph
            new_order_genes = [ordered_genes[i] for i in new_order]
            return new_order_genes
        else:
            raise UserWarning(
                "length of argument new_order must be equal to either the number of"
                + " genes {} or the number of graph nodes {}! Received: {}".format(
                    len(self._genes), self.adj_matrix.shape[0], new_order
                )
            )

    def reorder(self, new_order: np.ndarray):
        self._genes = self.get_reordered(new_order)

    def change_matrix(self, adj_matrix: np.ndarray):
        if topo.is_valid_adj_matrix(adj_matrix):
            self._genes = self.get_reordered(Genome.get_order(adj_matrix))
            self.adj_matrix = adj_matrix
        else:
            raise UserWarning("Invalid adj matrix!")

    @staticmethod
    def _gen_mutated(
        genes: list[dict],
        master: GenomeMaster = None,
        num_mut_p: float = 0.5,
        cat_mut_p: float = 0.05,
        power: float = 0.01,
        rng=None,
    ) -> list[dict]:
        """
        Generates a mutated version of genes encapsulated in a list[dict]
        :param genes: The genes object to make a mutated copy of.
        :param master: The GenomeMaster instance to use. If None, uses the default
            instance.
        :param num_mut_p: Probability of mutating numerical values.
            Must be in the interval [0, 1].
        :param cat_mut_p: Probability of mutating categorical values.
            Must be in the interval [0, 1].
        :param power: How much to change the old numerical values.
            See GenomeMaster.mutate_value() for more details.
        :param rng:
        :return: A copy of the genes object with mutated attributes.
        """
        if master is None:
            master = Genome.__default_genome_master
        if rng is None:
            rng = _default_rng
        output = deepcopy(genes)
        for gene in output:
            if gene["node_type"] == "sequential_functional":
                gene["contents"] = Genome._gen_mutated(
                    gene["contents"], master, num_mut_p, cat_mut_p, power, rng
                )
            else:
                for param in gene["modifiable_params"]:
                    p = rng.uniform()
                    if master.is_cat(gene["node_type"], param):
                        if p < cat_mut_p:
                            gene["config"][param] = master.gen_value(gene["node_type"], param)
                    elif master.is_num(gene["node_type"], param):
                        if p < num_mut_p:
                            gene["config"][param] = master.mutate_value(
                                gene["node_type"], param, gene["config"][param], power
                            )
                    else:
                        raise UserWarning(
                            "param {} of nodetype {}".format(param, gene["node_type"])
                            + " is neither numerical or categorical!"
                        )
        return output

    def gen_children(
        self,
        clones: int = 2,
        variants: int = 2,
        graph_mutations: int = 1,
        gr_mut_p=0.5,
        num_mut_p=0.5,
        cat_mut_p=0.05,
        mut_power=0.01,
        rng=None,
    ) -> list:
        """

        :param clones:
        :param variants:
        :param graph_mutations: How many point mutations each graph variant should have.
        :param gr_mut_p: Probability of mutating the model topology (the graph).
        :param num_mut_p: Probability of mutating numerical values.
            Must be in the interval [0, 1].
        :param cat_mut_p: Probability of mutating categorical values.
            Must be in the interval [0, 1].
        :param mut_power: How much to change the old numerical values.
            See GenomeMaster.mutate_value() for more details.
        :param rng:
        :return: A list of the generated Genome instances.
        """
        if clones < 0 or variants < 0:
            raise UserWarning(
                "The clones and variants arguments must be greater than 0! Received:"
                + " clones = {}, variants = {}".format(clones, variants)
            )
        if rng is None:
            rng = _default_rng
        output = []
        for i in range(clones):
            output.append(deepcopy(self))

        num_graph_variants = (rng.uniform(0, 1, variants) < gr_mut_p).sum()

        if num_graph_variants > 0:
            variant_graphs = topo.gen_mutated(
                arr=self.adj_matrix,
                n=num_graph_variants,
                mutations=graph_mutations,
                precomputed_valid=self.precomp,
            )
            if variant_graphs.shape[0] < num_graph_variants:
                warnings.warn(
                    "Number of graph variants is less than requested!"
                    + " Generated: {}; Requested: {}".format(variant_graphs.shape[0], variants)
                )
            for i in range(variant_graphs.shape[0]):
                output.append(
                    Genome(
                        input_layer=self.input_layer,
                        output_layer=self.output_layer,
                        genes=Genome._gen_mutated(
                            genes=self.get_reordered(Genome.get_order(variant_graphs[i])),
                            master=self.master,
                            num_mut_p=num_mut_p,
                            cat_mut_p=cat_mut_p,
                            power=mut_power,
                            rng=rng,
                        ),
                        adj_matrix=variant_graphs[i],
                        adj_mat_check=False,
                        master=self.master,
                    )
                )
        for i in range(variants - num_graph_variants):
            output.append(
                Genome(
                    input_layer=self.input_layer,
                    output_layer=self.output_layer,
                    genes=Genome._gen_mutated(
                        genes=self._genes,
                        master=self.master,
                        num_mut_p=num_mut_p,
                        cat_mut_p=cat_mut_p,
                        power=mut_power,
                        rng=rng,
                    ),
                    adj_matrix=self.adj_matrix,
                    adj_mat_check=False,
                    master=self.master,
                )
            )

        return output


if __name__ == "__main__":
    # from memory_profiler import profile

    # @profile
    def main(argv, *args):
        foo = Genome(keras.Input(10), keras.layers.Dense(1))
        m = foo.gen_model()
        print(m.summary())

    sys.exit(main(sys.argv))
