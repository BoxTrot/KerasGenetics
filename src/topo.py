import itertools
import os
import pickle
import timeit
import warnings
from math import factorial

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from tensorflow import keras


def batch_yield(it, batch=1000, raise_stop=False):
    out = []
    for i in range(batch):
        try:
            out.append(next(it))
        except StopIteration:
            if i == 0 and raise_stop:
                raise StopIteration
            else:
                return out
    return out


def combination(n, k):
    if n >= k:
        return factorial(n) / (factorial(k) * factorial(n - k))
    else:
        raise UserWarning("n must be greater than or equal to k!")


def array_bin_pos(n):
    stop = 2**n
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
    arr: np.ndarray, in_node: int = 0, out_node: int = -1, warning_lvl: int = 1
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
            return False
        else:
            raise UserWarning("Malformed array passed wih shape {}!".format(arr.shape))

    n = arr.shape[0]
    in_ind = np.arange(n)[in_node]
    out_ind = np.arange(n)[out_node]
    if _is_valid_adj_matrix_base(arr, in_node, out_node):
        gra: ig.Graph = ig.Graph.Adjacency(arr)
        if gra.is_dag():
            spaths = gra.get_all_simple_paths(in_ind, out_ind)
            truths = [[i in spaths[j] for j in range(len(spaths))] for i in range(n)]
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


def breed_adj_matrix(a: np.ndarray, b: np.ndarray, precomputed_valid=None) -> np.ndarray:
    if type(precomputed_valid) is str:
        if precomputed_valid.endswith(".pickle"):
            with open(precomputed_valid, "rb") as f:
                precomp = pickle.load(f)
                if type(precomp) is not np.ndarray:
                    raise UserWarning(
                        "Pickle {} did not contain a numpy ndarray".format(precomputed_valid)
                    )
        else:
            raise UserWarning("unknown file extension specified in {}".format(precomputed_valid))
    elif type(precomputed_valid) is np.ndarray:
        precomp: np.ndarray = precomputed_valid
    elif precomputed_valid is not None:
        raise UserWarning(
            "unknown variable type {} for argument precomputed_valid".format(
                type(precomputed_valid)
            )
        )

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


def average_pos(arr, in_node=0, out_node=-1, exclude_in_out=False):
    n = arr.shape[0]
    _in = [i for i in range(n)][in_node]
    _out = [i for i in range(n)][out_node]
    gr = ig.Graph.Adjacency(arr)
    pos = [[] for i in range(n)]
    paths = gr.get_all_simple_paths(_in, _out)
    for p in paths:
        for i in range(arr.shape[0]):
            try:
                pos[i].append(p.index(i))
            except ValueError:
                pass
    avgpos = [sum(j) / min(1, len(j)) for j in pos]
    if exclude_in_out:
        inds = [i for i in range(n)]
        inds.remove(_in)
        inds.remove(_out)
        return [avgpos[i] for i in inds]
    else:
        return avgpos


def gen_coords(n, in_node: int = 0, out_node: int = -1) -> np.ndarray:
    _in = np.arange(n)[in_node]
    _out = np.arange(n)[out_node]
    m = np.array(
        [arr.flatten() for arr in np.meshgrid(np.arange(n), np.arange(n), indexing="xy")]
    ).transpose()
    m = m[(m[:, 0] != m[:, 1])]
    m = m[(m[:, 0] != _out)]
    m = m[(m[:, 1] != _in)]
    return m


def gen_mutated(
    arr: np.ndarray,
    n: int = 1,
    mutations: int = 1,
    in_node: int = 0,
    out_node: int = -1,
    precomputed_valid=None,
    rng=np.random.default_rng(),
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
    :return: An ndarray containing all the mutant graphs. Access each one via the first
        array axis.
    """
    # if type(precomputed_valid) is str:
    #     if precomputed_valid.endswith(".pickle"):
    #         with open(precomputed_valid, "rb") as f:
    #             precomp = pickle.load(f)
    #             if type(precomp) is not np.ndarray:
    #                 raise UserWarning(
    #                     "Pickle {} did not contain a numpy ndarray".format(
    #                         precomputed_valid
    #                     )
    #                 )
    #     else:
    #         raise UserWarning(
    #             "unknown file extension specified in {}".format(precomputed_valid)
    #         )
    # elif type(precomputed_valid) is np.ndarray:
    #     precomp: np.ndarray = precomputed_valid
    # elif precomputed_valid is not None:
    #     raise UserWarning(
    #         "unknown variable type {} for argument precomputed_valid".format(
    #             type(precomputed_valid)
    #         )
    #     )
    if mutations < 1:
        raise UserWarning("Mutations argument must be at least 1!")
    if n < 1:
        raise UserWarning("N argument must be at least 1!")

    width = arr.shape[0]
    vals = width**2 - width * 2 + 1
    in_ind = np.arange(width)[in_node]
    out_ind = np.arange(width)[out_node]

    # if precomputed_valid is not None:
    #     if True not in [(i == arr).all() for i in precomp]:
    #         if not is_valid_adj_matrix(arr, in_ind, out_ind):
    #             raise UserWarning("arr is not a valid adj matrix!")
    if not is_valid_adj_matrix(arr, in_ind, out_ind):
        raise UserWarning("arr is not a valid adj matrix!")

    coords = rng.permutation(gen_coords(width, in_node, out_node))
    tried = []
    mutated = []
    if mutations == 1:
        for indices in coords:
            attempt = arr.copy("K")
            attempt[indices[0], indices[1]] = not attempt[indices[0], indices[1]]
            # if precomputed_valid is not None:
            #     if True in [(i == attempt).all() for i in precomp]:
            #         mutated.append(attempt)
            #     elif is_valid_adj_matrix(attempt, in_ind, out_ind):
            #         mutated.append(attempt)
            if is_valid_adj_matrix(attempt, in_ind, out_ind):
                mutated.append(attempt)
            if len(mutated) >= n:
                break
    else:
        num_poss = combination(width - 1, mutations)
        if num_poss < 10000:
            combs = rng.permutation(np.array(list(itertools.combinations(coords, mutations))), 0)
            for indices_group in combs:
                attempt = arr.copy("K")
                for indices in indices_group:
                    attempt[indices[0], indices[1]] = not attempt[indices[0], indices[1]]
                # if precomputed_valid is not None:
                #     if True in [(i == attempt).all() for i in precomp]:
                #         mutated.append(attempt)
                #     elif is_valid_adj_matrix(attempt, in_ind, out_ind):
                #         mutated.append(attempt)
                if is_valid_adj_matrix(attempt, in_ind, out_ind):
                    mutated.append(attempt)
                if len(mutated) >= n:
                    break

        else:
            it = itertools.combinations(coords, mutations)
            batch = rng.permutation(batch_yield(it))
            while batch.shape[0] > 0:
                for indices_group in batch:
                    attempt = arr.copy("K")
                    for indices in indices_group:
                        attempt[indices[0], indices[1]] = not attempt[indices[0], indices[1]]
                    # if precomputed_valid is not None:
                    #     if True in [(i == attempt).all() for i in precomp]:
                    #         mutated.append(attempt)
                    #     elif is_valid_adj_matrix(attempt, in_ind, out_ind):
                    #         mutated.append(attempt)
                    if is_valid_adj_matrix(attempt, in_ind, out_ind):
                        mutated.append(attempt)
                    if len(mutated) >= n:
                        break
                if len(mutated) >= n:
                    break
                batch = rng.permutation(batch_yield(it))
    return np.array(mutated)


def iterate_adj_matrices(num_nodes: int, padding: int = 1) -> np.ndarray:
    val_width = num_nodes - padding
    num_vals = val_width**2
    start = 2 ** (val_width * (val_width - 1))
    stop = 2**num_vals  # exclusive
    # print(start, stop)
    for n in range(start, stop):
        seq = np.array([i for i in map(int, np.binary_repr(n, num_vals))]).reshape(
            (val_width, val_width)
        )
        if one_in_every_row_col(seq, val_width):
            foo = np.zeros((num_nodes, num_nodes), dtype="B")
            foo[0:val_width, padding:] = seq
            yield foo


def gen_all_graphs(
    num_nodes: int, include_isomorphs: bool = False, temp_dir: str = None
) -> np.ndarray:
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
    :param include_isomorphs: Whether to include isomorphs in the list.
    :return: A numpy ndarray with shape (graphs, num_nodes, num_nodes).
        The in node is at index 0 in the graph, and the out node is at num_nodes-1.
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
            if out_node == num_nodes - 1 and in_node == 0:
                gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
                if gr.is_dag():
                    spaths = gr.get_all_simple_paths(in_node, out_node)
                    truths = [
                        [i in spaths[j] for j in range(len(spaths))] for i in range(num_nodes)
                    ]
                    if False not in [True in i for i in truths]:
                        # i.e. that for each node, there is a path from the in node to the
                        # out node that passes through it
                        if len(graphs) > 0:
                            isomorph = False
                            if not include_isomorphs:
                                for obj in graph_objects:
                                    if gr.isomorphic(obj):
                                        isomorph = True
                                        break
                                if not isomorph:
                                    graphs.append(adj_matrix)
                                    graph_objects.append(gr)
                                    print(len(graphs))
                            else:
                                graphs.append(adj_matrix)
                                graph_objects.append(gr)
                                print(len(graphs))
                        else:
                            graphs.append(adj_matrix)
                            graph_objects.append(gr)
                            print(len(graphs))
                    if temp_dir is not None:
                        temparray = np.array(graphs)
                        os.makedirs(temp_dir)
                        with open(os.path.join(temp_dir, "temp.pickle"), "wb") as f:
                            pickle.dump(temparray, f)
    if temp_dir is not None:
        temparray = np.array(graphs)
        os.makedirs(temp_dir)
        with open(os.path.join(temp_dir, "temp.pickle"), "wb") as f:
            pickle.dump(temparray, f)
    return np.array(graphs)


def gen_graph(num_nodes: int, rng=np.random.default_rng()) -> np.ndarray:
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
        adj_matrix = rng.integers(0, 2, (num_nodes, num_nodes), dtype="B")
        adj_matrix[:, 0] = 0
        adj_matrix[num_nodes - 1] = 0
        for i in range(num_nodes):
            adj_matrix[i, i] = 0
        if is_valid_adj_matrix(adj_matrix):
            break
    return adj_matrix


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
    # TODO: Change to use content from tf.keras.utils.serialize_keras_object
    gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
    num_nodes = gr.vcount()
    inputs = keras.Input(shape=input_shape, name="input")
    layer_list: list[keras.layers.Layer] = []
    output_config["name"] = "output"
    output_layer = keras.layers.Dense(0).from_config(output_config)
    for i in range(num_nodes - 2):
        layer_configs[i % len(layer_configs)]["name"] = "dense_{}".format(i)
        new_layer = keras.layers.Dense(0).from_config(layer_configs[i % len(layer_configs)])
        # new_layer.name = str(i)
        layer_list.append(new_layer)

    # repeatedly iterate across vertices array and add nodes whose inputs are already
    # in the array
    vertices = [
        None,
    ] * (num_nodes - 1)
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


def __apply_functional(layer_list: list, input):
    if len(layer_list) == 0:
        raise UserWarning("layer_list is empty!")
    last_output = input
    for i in range(len(layer_list)):
        if isinstance(layer_list[i], list):
            last_output = __apply_functional(layer_list[i], last_output)
        else:
            last_output = layer_list[i](last_output)
    return last_output


def model_from_graph_and_layers(
    adj_matrix: np.ndarray,
    in_node: int,
    out_node: int,
    input_layer,
    layer_list: list,
    output_layer,
) -> keras.Model:
    gr: ig.Graph = ig.Graph.Adjacency(adj_matrix)
    num_nodes = gr.vcount()

    # repeatedly iterate across vertices array and add nodes whose inputs are already
    # in the array
    vertices = [
        None,
    ] * (num_nodes - 1)
    vertices[in_node] = input_layer
    while None in vertices:
        for i in range(1, num_nodes - 1):
            if vertices[i] is None:
                in_inds = adj_matrix[:, i].nonzero()[0].tolist()
                if len(in_inds) > 0:
                    if not (True in [vertices[ind] is None for ind in in_inds]):
                        if len(in_inds) == 1:
                            vertices[i] = __apply_functional(
                                [layer_list[i - 1]], vertices[in_inds[0]]
                            )
                        else:
                            x = keras.layers.concatenate([vertices[j] for j in in_inds])
                            vertices[i] = __apply_functional([layer_list[i - 1]], x)

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
    setup_1 = "import pickle\nimport numpy as np\nrng=np.random.default_rng()\narrs = rng.integers(0, 2, size=(1000, 6, 6))\narrs[:, -1, :] = 0\narrs[:, :, 0] = 0"
    stmt_1 = "for i in arrs:\n    is_valid_adj_matrix(i)"
    setup_2 = "\nwith open('all_6_node_graphs_isomorphs.pickle', 'rb') as f:\n    precomp = pickle.load(f)"
    stmt_2 = "for j in arrs:\n    arr_match(j, precomp)"
    print(timeit.timeit(stmt_1, setup=setup_1, globals=globals(), number=10), flush=True)
    print(
        timeit.timeit(stmt_2, setup=setup_1 + setup_2, globals=globals(), number=10),
        flush=True,
    )


if __name__ == "__main__":
    main()
