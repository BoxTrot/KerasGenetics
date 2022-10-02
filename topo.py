import pickle
import warnings
from math import factorial

import igraph as ig
import keras
import keras.callbacks
import keras.layers
import keras.losses
import keras.metrics
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


def array_bin_pos(n):
    stop = 2 ** n
    out = np.array([[int(i) for i in np.binary_repr(j, n)] for j in range(0, stop)])
    return out


def iter_bin_pos(n):
    out = array_bin_pos(n)
    for row in range(out.shape[0]):
        yield out[row]


@njit
def duped_rows(arr: np.ndarray):
    for i in prange(arr.shape[0]):
        for j in prange(arr.shape[0]):
            if (i != j) and (arr[i] == arr[j]).all():
                return True
    return False


@njit
def one_in_every_row_col(seq: np.ndarray, width: int):
    for i in range(width):
        if 1 not in seq[i]:
            # one_in_every_row = False
            return False
        if 1 not in seq[:, i]:
            # one_in_every_col = False
            return False
    return True


@njit
def _is_valid_adj_matrix_base(
        arr: np.ndarray,
        in_node: int = 0,
        out_node: int = -1,
) -> bool:
    n = arr.shape[0]

    for i in range(n):
        if arr[i, i] != 0:
            # i.e. if arr[i, i] == 1, the node has an edge that points to itself,
            # creating a cycle
            return False

    outs = np.array([(arr[i] == 0).all() for i in range(n)])
    ins = np.array([(arr[:, i] == 0).all() for i in range(n)])
    if (not outs[out_node]) or (not ins[in_node]):
        return False
    if (np.count_nonzero(outs) != 1) or (np.count_nonzero(ins) != 1):
        return False
    return True


def is_valid_adj_matrix(
        arr: np.ndarray,
        in_node: int = 0,
        out_node: int = -1,
        warning_lvl: int = 1
) -> bool:
    """
    Requirements for the graph represented by the adjacency matrix:
        - Directed acyclic graph.
        - Weakly connected.
        - There is exactly one in node and one out node, at the positions identified in
          the arguments.
        - For every node that isn't the in or out node, a path exists from the in node
          to the out node that passes through it.
    :param arr: The adjacency matrix.
    :param in_node: The row and column of the in node.
    :param out_node: The row and column of the out node.
    :param warning_lvl: The warning level. 0 for no warnings, 1 for malformed arrays
                        (not a matrix, or not square), 2 to raise errors for
                        malformed arrays. Any value other than 0, 1, or 2 is treated as
                        2.
    :rtype: bool
    """
    if (len(arr.shape) != 2) or (arr.shape[0] != arr.shape[1]):
        if warning_lvl == 0:
            return False
        elif warning_lvl == 1:
            warnings.warn("Malformed array passed wih shape {}!".format(arr.shape))
        else:
            raise UserWarning(
                "Malformed array passed wih shape {}!".format(arr.shape)
            )

    n = arr.shape[0]
    in_ind = np.arange(n)[in_node]
    out_ind = np.arange(n)[out_node]
    if _is_valid_adj_matrix_base(arr, in_node, out_node):
        gra: ig.Graph = ig.Graph.Adjacency(arr)
        if gra.is_dag():
            spaths = gra.get_all_simple_paths(in_ind, out_ind)
            truths = [[i in spaths[j] for j in range(len(spaths))] for i in
                      range(n)]
            if False not in [True in i for i in truths]:
                # i.e. that for each node, there is a path from the in node to the
                # out node that passes through it
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def breed_adj_matrix(
        a: np.ndarray, b: np.ndarray, precomputed_valid=None
) -> np.ndarray:
    if type(precomputed_valid) is str:
        if precomputed_valid.endswith(".pickle"):
            with open(precomputed_valid, "rb") as f:
                precomp = pickle.load(f)
                if type(precomp) is not np.ndarray:
                    raise UserWarning(
                        "Pickle {} did not contain a numpy ndarray".format(
                            precomputed_valid))
        else:
            raise UserWarning(
                "unknown file extension specified in {}".format(precomputed_valid))
    elif type(precomputed_valid) is np.ndarray:
        precomp: np.ndarray = precomputed_valid
    elif precomputed_valid is not None:
        raise UserWarning(
            "unknown variable type {} for argument precomputed_valid".format(
                type(precomputed_valid)))

    avg: np.ndarray = (a + b) / 2
    points: np.ndarray = np.argwhere(avg == 0.5)
    out = []
    for toggle in iter_bin_pos(points.shape[0]):
        attempt = avg.copy("K")
        for i in range(points.shape[0]):
            pos = points[i]
            attempt[pos[0], pos[1]] = toggle[i]
        if precomputed_valid is not None:
            if True in [(i == attempt).all() for i in precomp]:
                out.append(attempt)
        else:
            if is_valid_adj_matrix(attempt):
                out.append(attempt)
    return np.array(out)


def gen_mutated(
        arr: np.ndarray,
        n: int = 1,
        mutations: int = 1,
        in_node: int = 0,
        out_node: int = -1,
        precomputed_valid=None,
        rng=np.random.default_rng()
) -> np.ndarray:
    """
    Taking a base graph, creates new graphs with "point mutations" in the adjacency
    matrix.
    :param arr: Base graph to use.
    :param n: How many unique mutated graphs to generate.
    :param mutations: How many
    :param in_node:
    :param out_node:
    :param precomputed_valid:
    :param rng:
    :return:
    """
    if type(precomputed_valid) is str:
        if precomputed_valid.endswith(".pickle"):
            with open(precomputed_valid, "rb") as f:
                precomp = pickle.load(f)
                if type(precomp) is not np.ndarray:
                    raise UserWarning(
                        "Pickle {} did not contain a numpy ndarray".format(
                            precomputed_valid))
        else:
            raise UserWarning(
                "unknown file extension specified in {}".format(precomputed_valid))
    elif type(precomputed_valid) is np.ndarray:
        precomp: np.ndarray = precomputed_valid
    elif precomputed_valid is not None:
        raise UserWarning(
            "unknown variable type {} for argument precomputed_valid".format(
                type(precomputed_valid)))
    if mutations < 1:
        raise UserWarning("Mutations argument must be at least 1!")
    if n < 1:
        raise UserWarning("N argument must be at least 1!")

    width = arr.shape[0]
    vals = (width ** 2 - width * 2 + 1)
    num_possible = factorial(vals) / (
            factorial(mutations) * factorial(vals - mutations))
    in_ind = np.arange(width)[in_node]
    out_ind = np.arange(width)[out_node]
    tried = []
    mutated = []
    while (len(mutated) < n) and (len(tried) < num_possible):
        indices = rng.integers(0, width, (mutations, 2))
        while (
                (indices[:, 0] == out_ind).any() or
                (indices[:, 1] == in_ind).any() or
                (True in [(indices == i).all() for i in tried]) or
                duped_rows(indices)
        ):
            indices = rng.integers(0, width, (mutations, 2))
        tried.append(indices)
        attempt = arr.copy("K")
        for ind in indices:
            attempt[ind[0], ind[1]] = not attempt[ind[0], ind[1]]
        if precomputed_valid is not None:
            if True in [(i == attempt).all() for i in precomp]:
                mutated.append(attempt)
        elif is_valid_adj_matrix(attempt, in_ind, out_ind):
            mutated.append(attempt)
    return np.array(mutated)


@njit
def iterate_adj_matrices(num_nodes: int, padding: int = 1) -> np.ndarray:
    val_width = num_nodes - padding
    num_vals = val_width ** 2
    start = 2 ** (val_width * (val_width - 1))
    stop = 2 ** num_vals  # exclusive
    print(start, stop)
    for n in range(start, stop):
        seq = np.array([i for i in map(int, np.binary_repr(n, num_vals))]).reshape(
            (val_width, val_width))
        if one_in_every_row_col(seq, val_width):
            foo = np.zeros((num_nodes, num_nodes))
            foo[0:val_width, padding:] = seq
            yield foo


def gen_all_graphs(num_nodes: int) -> list[tuple[np.ndarray, int, int]]:
    """
    Generates a list of all possible directed graphs that meet the following criteria:
        - Weakly connected
        - Acyclic
        - Exactly one "in node", which only has outgoing edges
        - Exactly one "out node", which only has incoming edges
        - For each node/vertex, there is a simple path from the "in node" to the
          "out node" that passes through it
    :param num_nodes: How many nodes or vertices should be in the graph, including the
        in node and out note. Must be a positive integer.
    :return: A list of tuples. Each tuple contains, in order, the adjacency matrix, the
        number of the in node, and the number of the out node. The in node should be 0
        and the out node should be num_nodes - 1 in all cases, but the node numbers are
        provided as a fallback.
    """
    graphs = []
    graph_objects = []
    for adj_matrix in iterate_adj_matrices(num_nodes, 1):
        for i in range(num_nodes):
            adj_matrix[i, i] = 0
        outs = [(adj_matrix[i] == 0).all() for i in range(num_nodes)]
        ins = [(adj_matrix[:, i] == 0).all() for i in range(num_nodes)]
        if (outs.count(True) == 1) and (ins.count(True) == 1):
            out_node = outs.index(True)
            in_node = ins.index(True)
            gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
            if gr.is_dag():
                spaths = gr.get_all_simple_paths(in_node, out_node)
                truths = [[i in spaths[j] for j in range(len(spaths))] for i in
                          range(num_nodes)]
                if False not in [True in i for i in truths]:
                    # i.e. that for each node, there is a path from the in node to the
                    # out node that passes through it
                    if len(graphs) > 0:
                        isomorph = False
                        for obj in graph_objects:
                            if gr.isomorphic(obj):
                                isomorph = True
                                break
                        if not isomorph:
                            graphs.append((adj_matrix, in_node, out_node))
                            graph_objects.append(gr)
                            print(len(graphs))
                    else:
                        graphs.append((adj_matrix, in_node, out_node))
                        graph_objects.append(gr)
                        print(len(graphs))
    return graphs


def gen_graph(
        num_nodes: int, rng=np.random.default_rng()
) -> tuple[np.ndarray, int, int]:
    """
    Generates a random directed acyclic graph that meets the following criteria:
        - Weakly connected
        - Exactly one "in node", which only has outgoing edges
        - Exactly one "out node", which only has incoming edges
        - For each node/vertex, there is a simple path from the "in node" to the
          "out node" that passes through it
    :param num_nodes: How many nodes or vertices should be in the graph, including the
        in node and out note. Must be a positive integer.
    :param rng: The random number generator. Default is the numpy default rng.
    :return: A tuple. It contains, in order, the adjacency matrix, the
        number of the in node, and the number of the out node. The in node should be 0
        and the out node should be num_nodes - 1 in all cases, but the node numbers are
        provided as a fallback.
    """
    # generate a directed acyclic graph with one entry and one exit point
    tries = 0
    while True:
        tries += 1
        adj_matrix = rng.integers(0, 2, (num_nodes, num_nodes))
        adj_matrix[:, 0] = 0
        adj_matrix[num_nodes - 1] = 0
        for i in range(num_nodes):
            adj_matrix[i, i] = 0
        outs = [(adj_matrix[i] == 0).all() for i in range(num_nodes)]
        ins = [(adj_matrix[:, i] == 0).all() for i in range(num_nodes)]
        # print(outs.count(True), ins.count(True))
        if (outs.count(True) == 1) and (ins.count(True) == 1):
            out_node = outs.index(True)
            in_node = ins.index(True)
            gr = ig.Graph.Adjacency(adj_matrix)
            if gr.is_connected("weak") and gr.is_dag():
                spaths = gr.get_all_simple_paths(in_node, out_node)
                truths = [[i in spaths[j] for j in range(len(spaths))] for i in
                          range(num_nodes)]
                if False not in [True in i for i in truths]:
                    # i.e. that for each node, there is a path from the in node to the
                    # out node that passes through it
                    print("Tried {} times to get this graph.".format(tries))
                    break
    return adj_matrix, in_node, out_node


def plot_graph(adj_matrix: np.ndarray, in_node: int, out_node: int):
    gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
    num_nodes = adj_matrix.shape[0]
    fig, ax = plt.subplots()
    vertex_colors = ["steelblue"] * num_nodes
    vertex_colors[in_node] = "limegreen"
    vertex_colors[out_node] = "gold"
    vertex_labels = [str(i) for i in range(num_nodes)]
    vertex_labels[in_node] += "_in"
    vertex_labels[out_node] += "_out"
    ig.plot(
        gr,
        target=ax,
        vertex_size=20,
        vertex_color=vertex_colors,
        vertex_label=vertex_labels,
        edge_width=3,
    )
    plt.show()


def model_from_graph_and_configs(
        adj_matrix: np.ndarray,
        in_node: int,
        out_node: int,
        input_shape: tuple,
        layer_configs: list[dict],
        output_config: dict,
):
    gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
    num_nodes = gr.vcount()
    inputs = keras.Input(shape=input_shape, name="input")
    layer_list: list[keras.layers.Layer] = []
    output_config["name"] = "output"
    output_layer = keras.layers.Dense(0).from_config(output_config)
    for i in range(num_nodes - 2):
        layer_configs[i % len(layer_configs)]["name"] = "dense_{}".format(i)
        new_layer = keras.layers.Dense(0).from_config(
            layer_configs[i % len(layer_configs)]
        )
        # new_layer.name = str(i)
        layer_list.append(new_layer)

    # repeatedly iterate across vertices array and add nodes whose inputs are already
    # in the array
    vertices = [None, ] * (num_nodes - 1)
    vertices[in_node] = inputs
    while None in vertices:
        for i in range(1, num_nodes - 1):
            if vertices[i] is None:
                in_inds = adj_matrix[:, i].nonzero()[0].tolist()
                if len(in_inds) > 0:
                    if not (True in [vertices[ind] is None for ind in in_inds]):
                        if len(in_inds) == 1:
                            vertices[i] = layer_list[i - 1](vertices[in_inds[0]])
                        else:
                            x = keras.layers.concatenate([vertices[j] for j in in_inds])
                            vertices[i] = layer_list[i - 1](x)

    output_ins = adj_matrix[:, out_node].nonzero()[0].tolist()
    if len(output_ins) == 1:
        output = output_layer(vertices[output_ins[0]])
    else:
        z = keras.layers.concatenate([vertices[j] for j in output_ins])
        output = output_layer(z)
    model = keras.Model(
        inputs=inputs,
        outputs=output,
    )
    return model


def model_from_graph_and_layers(
        adj_matrix: np.ndarray,
        in_node: int,
        out_node: int,
        input_layer,
        layer_list: list,
        output_layer,
):
    gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
    num_nodes = gr.vcount()

    # repeatedly iterate across vertices array and add nodes whose inputs are already
    # in the array
    vertices = [None, ] * (num_nodes - 1)
    vertices[in_node] = input_layer
    while None in vertices:
        for i in range(1, num_nodes - 1):
            if vertices[i] is None:
                in_inds = adj_matrix[:, i].nonzero()[0].tolist()
                if len(in_inds) > 0:
                    if not (True in [vertices[ind] is None for ind in in_inds]):
                        if len(in_inds) == 1:
                            vertices[i] = layer_list[i - 1](vertices[in_inds[0]])
                        else:
                            x = keras.layers.concatenate([vertices[j] for j in in_inds])
                            vertices[i] = layer_list[i - 1](x)

    output_ins = adj_matrix[:, out_node].nonzero()[0].tolist()
    if len(output_ins) == 1:
        output = output_layer(vertices[output_ins[0]])
    else:
        z = keras.layers.concatenate([vertices[j] for j in output_ins])
        output = output_layer(z)
    model = keras.Model(
        inputs=input_layer,
        outputs=output,
    )
    return model


def main():
    num_nodes = 6
    rng = np.random.default_rng()
    foo = gen_graph(6)[0]
    print(foo)
    print(gen_mutated(foo, 5, 2))
    # all_g = gen_all_graphs(num_nodes)
    # with open("all_6_node_graphs.pickle", "xb") as f:
    #     pickle.dump(all_g, f)
    # print(len(all_g))
    # print(False in [(g[1] == 0 and g[2] == num_nodes - 1) for g in all_g])

    # gr, adj_matrix, in_node, out_node = gen_graph(num_nodes)
    # plot_graph(gr, adj_matrix, in_node, out_node)
    #
    # foo = keras.layers.Dense(10, "swish").get_config()
    #
    # model = model_from_graph(gr, adj_matrix, in_node, out_node, (10,), [foo], foo)
    # print(model.summary())


if __name__ == '__main__':
    main()
