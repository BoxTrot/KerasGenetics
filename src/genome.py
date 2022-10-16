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
__node_types = {
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
__valid_node_types = __node_types.keys()


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
    def __init__(self, rng=np.random.default_rng()):
        self.master = {
            "Dense": {
                "units": {
                    "type": "int",
                    "range": (0, 400),
                    "func": lambda: rng.integers(0, 400, endpoint=True),
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
                    "func": lambda: rng.uniform(0.0, 0.7),
                },
            },
            "AlphaDropout": {
                "rate": {
                    "type": "float",
                    "range": (0.0, 0.7),
                    "func": lambda: rng.uniform(0.0, 0.7),
                },
            },
        }


class Genome:
    def __init__(
        self,
        input_layer: keras.layers.InputLayer,
        output_layer: keras.layers.Layer,
        genes: list[dict] = None,
        adj_matrix: np.ndarray = None,
    ):
        if genes is None:
            self.__genes = [
                {
                    "node_type": "Dense",
                    "config": {
                        "name": "dense_0",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 1,
                        "activation": "linear",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "GlorotUniform",
                            "config": {"seed": None},
                        },
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                    "modifiable_params": ["units", "activation", "kernel_initializer"],
                },
                {
                    "node_type": "Dense",
                    "config": {
                        "name": "dense_1",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 1,
                        "activation": "linear",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "GlorotUniform",
                            "config": {"seed": None},
                        },
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                    "modifiable_params": ["units", "activation", "kernel_initializer"],
                },
                {
                    "node_type": "Dense",
                    "config": {
                        "name": "dense_2",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 1,
                        "activation": "linear",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "GlorotUniform",
                            "config": {"seed": None},
                        },
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                    "modifiable_params": ["units", "activation", "kernel_initializer"],
                },
            ]
        else:
            if self.check_gene_integrity(genes, 1):
                self.__genes = genes
            else:
                raise UserWarning(
                    "input argument genes was invalid! contents: {}".format(genes)
                )
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.adj_matrix = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )

    @staticmethod
    def check_gene_integrity(genes: list[dict], warnlevel: int = 1) -> bool:
        """

        :param genes:
        :param warnlevel: If any issues are found in the input, how to raise them if at all.
            0 = silent, 1 = warnings, 2 or any other value = exceptions
        :return: True if integrity good, otherwise False
        """
        global __valid_node_types
        for entry in genes:
            entry_keys = entry.keys()
            if "node_type" in entry_keys:
                if entry["node_type"] == "sequential_functional":
                    if "contents" in entry_keys:
                        if isinstance(entry["contents"], list):
                            if not Genome.check_gene_integrity(
                                entry["contents"], warnlevel
                            ):
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
                        warn_raise(
                            '"contents" not found in {}'.format(entry), warnlevel
                        )
                        return False
                elif entry["node_type"] in __valid_node_types:
                    if "config" not in entry_keys:
                        warn_raise('"config" not found in {}'.format(entry), warnlevel)
                        return False
                    elif not isinstance(entry["config"], dict):
                        warn_raise(
                            '"config" in {} is not a dict!'.format(entry), warnlevel
                        )
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
    def __layers_from_genes(genes: list[dict]):
        global __valid_node_types, __node_types
        output = []
        for entry in genes:
            if entry["node_type"] == "sequential_functional":
                content = Genome.__layers_from_genes(entry["contents"])
                x = content[0]
                for i in range(1, len(content)):
                    x = content[i](x)
                output.append(x)
            elif entry["node_type"] in __valid_node_types:
                if __node_types[entry["node_type"]]["default"] is None:
                    output.append(
                        __node_types[entry["node_type"]]["class"]().from_config(
                            entry["config"]
                        )
                    )
                else:
                    output.append(
                        __node_types[entry["node_type"]]["class"](
                            *__node_types[entry["node_type"]]["default"]
                        ).from_config(entry["config"])
                    )
        return output

    def layers_from_genes(self):
        return Genome.__layers_from_genes(self.__genes)


def main(argv, *args):
    rng = np.random.default_rng()
    modifiable_params_basis = {
        "Dense": {
            "units": {
                "type": "int",
                "range": (0, 400),
                "func": lambda: rng.integers(0, 400, endpoint=True),
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
                "func": lambda: rng.uniform(0.0, 0.7),
            },
        },
        "AlphaDropout": {
            "rate": {
                "type": "float",
                "range": (0.0, 0.7),
                "func": lambda: rng.uniform(0.0, 0.7),
            },
        },
    }
    foo = {
        "name": "dense",
        "trainable": True,
        "dtype": "float32",
        "units": 1,
        "activation": "linear",
        "use_bias": True,
        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
        "bias_initializer": {"class_name": "Zeros", "config": {}},
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }
    genes = [
        {
            "node_type": "Dense",
            "config": {
                "name": "dense_0",
                "trainable": True,
                "dtype": "float32",
                "units": 1,
                "activation": "linear",
                "use_bias": True,
                "kernel_initializer": {
                    "class_name": "GlorotUniform",
                    "config": {"seed": None},
                },
                "bias_initializer": {"class_name": "Zeros", "config": {}},
                "kernel_regularizer": None,
                "bias_regularizer": None,
                "activity_regularizer": None,
                "kernel_constraint": None,
                "bias_constraint": None,
            },
            "modifiable_params": ["units", "activation", "kernel_initializer"],
        },
        {
            "node_type": "Dense",
            "config": {
                "name": "dense_1",
                "trainable": True,
                "dtype": "float32",
                "units": 1,
                "activation": "linear",
                "use_bias": True,
                "kernel_initializer": {
                    "class_name": "GlorotUniform",
                    "config": {"seed": None},
                },
                "bias_initializer": {"class_name": "Zeros", "config": {}},
                "kernel_regularizer": None,
                "bias_regularizer": None,
                "activity_regularizer": None,
                "kernel_constraint": None,
                "bias_constraint": None,
            },
            "modifiable_params": ["units", "activation", "kernel_initializer"],
        },
        {
            "node_type": "Dense",
            "config": {
                "name": "dense_2",
                "trainable": True,
                "dtype": "float32",
                "units": 1,
                "activation": "linear",
                "use_bias": True,
                "kernel_initializer": {
                    "class_name": "GlorotUniform",
                    "config": {"seed": None},
                },
                "bias_initializer": {"class_name": "Zeros", "config": {}},
                "kernel_regularizer": None,
                "bias_regularizer": None,
                "activity_regularizer": None,
                "kernel_constraint": None,
                "bias_constraint": None,
            },
            "modifiable_params": ["units", "activation", "kernel_initializer"],
        },
    ]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
