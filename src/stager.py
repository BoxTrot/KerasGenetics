import json
import os
import pickle
import sys
import time
from typing import TypeAlias, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import agent
import genome
import preprocessing
import topo
import utils

"""
The class that will coordinate population initialisation, training, culling, and
reproduction.
Use asexual reproduction with mutation
"""

pandas_array: TypeAlias = Union[pd.DataFrame, pd.Series]

__default_rng = np.random.default_rng()


def gen_train_val_splits(
    train_val: pd.DataFrame,
    target: pandas_array,
    val_split: float = 0.8,
    num=100,
    rng=None,
    preproc_func: callable = preprocessing.transform_data,
    preprocess: bool = True,
) -> tuple[list[tf.Tensor], list[pandas_array], list[tf.Tensor], list[pandas_array]]:
    if rng is None:
        rng = __default_rng
    train_tensor_list = []
    train_target_list = []
    val_tensor_list = []
    val_target_list = []
    for i in range(num):
        inds = rng.permutation(train_val.shape[0])
        train_inds = inds[: int(train_val.shape[0] * val_split)]
        val_inds = inds[int(train_val.shape[0] * val_split) :]
        if preprocess:
            train_proc, val_proc = preproc_func(
                train_val.iloc[train_inds], train_val.iloc[val_inds]
            )
        else:
            train_proc = train_val.iloc[train_inds]
            val_proc = train_val.iloc[val_inds]
        train_tensor_list.append(tf.convert_to_tensor(train_proc, dtype=tf.float32))
        val_tensor_list.append(tf.convert_to_tensor(val_proc, dtype=tf.float32))
        train_target_list.append(target.iloc[train_inds])
        val_target_list.append(target.iloc[val_inds])
    return train_tensor_list, train_target_list, val_tensor_list, val_target_list


def train_val_tensor_iter(
    total: int,
    train_val: pd.DataFrame,
    target: pandas_array,
    val_split: float = 0.8,
    batch_size: int = 100,
    rng=None,
    preproc_func: callable = preprocessing.transform_data,
    preprocess: bool = True,
):
    if rng is None:
        rng = __default_rng
    (
        train_tensor_list,
        train_target_list,
        val_tensor_list,
        val_target_list,
    ) = gen_train_val_splits(
        train_val, target, val_split, batch_size, rng, preproc_func, preprocess
    )
    for total_ind in range(total):
        batch_ind = total_ind % batch_size
        if batch_ind == 0 and total_ind != 0:
            (
                train_tensor_list,
                train_target_list,
                val_tensor_list,
                val_target_list,
            ) = gen_train_val_splits(
                train_val, target, val_split, batch_size, rng, preproc_func, preprocess
            )
        yield (
            train_tensor_list[batch_ind],
            train_target_list[batch_ind],
            val_tensor_list[batch_ind],
            val_target_list[batch_ind],
        )


def popgen(
    N,
    nodes,
    input_layer: keras.layers.InputLayer,
    output_layer: keras.layers.Layer,
) -> list[agent.Agent]:
    rng = np.random.default_rng()
    genome_base = genome.Genome(
        input_layer=input_layer,
        output_layer=output_layer,
        adj_matrix=np.eye(nodes, k=1, dtype="B"),
    )
    output = [agent.Agent(genome_base)]
    output.extend(
        output[0].gen_children(clones=0, variants=int(N / 4), graph_mutations=1, cat_mut_p=0.0)
    )
    output.extend(
        output[0].gen_children(
            clones=0,
            variants=int(N / 4),
            graph_mutations=2,
            num_mut_p=1.0,
            cat_mut_p=0.0,
            mut_power=0.05,
        )
    )
    nextind = len(output)
    gr_muts = 2
    while len(output) < N:
        output.extend(
            output[0].gen_children(
                clones=0,
                variants=N - len(output),
                graph_mutations=gr_muts,
                num_mut_p=1.0,
                cat_mut_p=0.0,
                mut_power=0.05,
            )
        )
        gr_muts += 1
    if nodes == 5:
        for i in range(nextind, N):
            try:
                output[i].genes.change_matrix(
                    utils.all_5_node[rng.integers(0, utils.all_5_node.shape[0])]
                )
            except Exception as err:
                print(i, N)
                raise err
    elif nodes == 6:
        for i in range(nextind, N):
            output[i].genes.change_matrix(
                utils.all_6_node[rng.integers(0, utils.all_6_node.shape[0])]
            )
    else:
        for i in range(nextind, N):
            output[i].genes.change_matrix(topo.gen_graph(nodes))
    return output


class Stager:
    def __init__(
        self,
    ):
        self.time_format = "(%d %b %Y %H:%M:%S)"

    def str_time(self):
        return time.strftime(self.time_format)

    def run(
        self,
        *,
        train_val: pd.DataFrame,
        train_val_target: pd.Series,
        nodes: int = 5,
        pop_size: int = 100,
        generations: int = 100,
        optimizer: keras.optimizers.Optimizer = None,
        loss_metric: keras.losses.Loss = None,
        model_epochs=70,
        trials_per_generation: int = 1,
        keep_fraction: float = 0.4,
        reprod_fraction: float = 0.2,
        child_clones: int = 2,
        val_split: float = 0.8,
        preproc_func: callable = None,
        rng=None,
        save: int = 5,
        save_dir: str = "",
        flush: bool = True,
        verbose: int = 2,
    ):
        # Default argument values
        if rng is None:
            rng = np.random.default_rng()
        if loss_metric is None:
            loss_metric = keras.losses.BinaryCrossentropy()
        if optimizer is None:
            optimizer = keras.optimizers.Adadelta(1.0)
        if preproc_func is None:
            preproc_func = preprocessing.transform_data

        if verbose > 1:
            print("{} Starting stager run...".format(self.str_time()), flush=flush)

        # Check that the number of child and variant clones is a sensible value
        keep_num = round(pop_size * keep_fraction)
        reprod_num = round(pop_size * reprod_fraction)
        children_per_indiv = np.floor((pop_size - keep_num) / reprod_num)
        if children_per_indiv <= child_clones:
            raise UserWarning(
                "{} ".format(self.str_time())
                + "Num child clones is more than the number of possible children!"
                + "{} children vs {} clones".format(children_per_indiv, child_clones)
            )

        # Make save folders
        if (save_dir != "") and (save_dir != ".\\"):
            if verbose > 1:
                print("{} Making save folders...".format(self.str_time()), flush=flush)
            os.makedirs(save_dir, exist_ok=True)
        save_folder_name = "generation_{:0>" + str(len(str(generations))) + "}"

        batch_size = min(int(train_val.shape[0] * val_split), 1000)
        input_layer = keras.Input(train_val.shape[1])
        output_layer = keras.layers.Dense(1, activation="sigmoid")

        if verbose > 1:
            print("{} Creating initial population...".format(self.str_time()), flush=flush)
        pop = popgen(pop_size, nodes, input_layer, output_layer)
        if verbose > 1:
            print("{} Finished creating population.".format(self.str_time()), flush=flush)

        if verbose > 1:
            print("{} Creating train-val split generator...".format(self.str_time()), flush=flush)
        train_val_generator = train_val_tensor_iter(
            generations,
            train_val,
            train_val_target,
            val_split,
            rng=rng,
            preproc_func=preproc_func,
        )
        if verbose > 1:
            print("{} Finished creating generator.".format(self.str_time()), flush=flush)

        fitness = np.zeros((len(pop),))
        for gen_num in range(generations):
            print("{} Starting generation {}...".format(self.str_time(), gen_num + 1))
            train_tensor: tf.Tensor
            train_target: pandas_array
            val_tensor: tf.Tensor
            val_target: pandas_array
            try:
                train_tensor, train_target, val_tensor, val_target = next(train_val_generator)
            except StopIteration as err:
                print(
                    "{} Iteration stopped early! Was at generation {}".format(
                        self.str_time(), gen_num
                    )
                )
                raise err
            fitness = np.zeros((len(pop),))
            histories = []
            if verbose > 2:
                print("{} Running individual ".format(self.str_time()), end="")
            for i in range(len(pop)):
                if verbose > 2:
                    print(i, end=" ")
                pop[i].gen_model()
                pop[i].model.compile(
                    optimizer=optimizer,
                    loss=loss_metric,
                )
                history: keras.callbacks.History = pop[i].model.fit(
                    train_tensor,
                    train_target,
                    batch_size,
                    model_epochs,
                    verbose=0,
                    validation_data=(val_tensor, val_target),
                )
                histories.append(history)
                fitness[i] = 1.0 / history.history["val_loss"][-1]
            if verbose > 2:
                print("")

            argsorted = np.flip(
                np.argsort(fitness, kind="stable")
            )  # order is index of greatest value first, index of smallest value last

            if verbose > 0:
                fitness_stats = {
                    "fitness_mean": fitness.mean(),
                    "fitness_std": fitness.std(),
                    "fitness_1st": fitness[argsorted[0]],
                    "fitness_2nd": fitness[argsorted[1]],
                    "fitness_3rd": fitness[argsorted[2]],
                    "fitness_min": fitness[argsorted[-1]],
                }
                print("{} Generation {} results -".format(self.str_time(), gen_num + 1), end="")
                for k, v in fitness_stats.items():
                    print(" {}: {};".format(k, v), end="")
                print("", flush=flush)

            if save > 0 and (gen_num % save) == 0:
                if verbose > 1:
                    print("{} Saving progress...".format(self.str_time()), flush=flush)
                gen_folder = save_folder_name.format(gen_num + 1)
                gen_save_path = os.path.join(save_dir, gen_folder)
                suffix = ""
                try:
                    os.makedirs(gen_save_path, exist_ok=False)
                except OSError:
                    while os.path.exists(gen_save_path):
                        if suffix == "":
                            suffix = "_01"
                        elif int(suffix[1:]) >= 99:
                            raise UserWarning("Too many similar folders!")
                        else:
                            suffix = "_{:0>2}".format(int(suffix[1:])+1)
                        gen_save_path = os.path.join(save_dir, gen_folder + suffix)
                os.makedirs(gen_save_path, exist_ok=True)
                with open(os.path.join(gen_save_path, "arguments.json"), "x", encoding="utf8") as f:
                    to_save_dict = {
                        "nodes": nodes,
                        "pop_size": pop_size,
                        "generations": generations,
                        "model_epochs": model_epochs,
                        "trials_per_generation": trials_per_generation,
                        "keep_fraction": keep_fraction,
                        "reprod_fraction": reprod_fraction,
                        "child_clones": child_clones,
                        "val_split": val_split,
                        "save": save,
                        "save_dir": save_dir,
                    }
                    json.dump(to_save_dict, f)
                    del to_save_dict
                with open(os.path.join(gen_save_path, "primitives.json"), "x", encoding="utf8") as f:
                    to_save_dict = {
                        "gen_num": gen_num,
                        "batch_size": batch_size,
                        "keep_num": keep_num,
                        "reprod_num": reprod_num,
                        "children_per_indiv": children_per_indiv,
                        "pop_len": len(pop),
                    }
                    json.dump(to_save_dict, f)
                    del to_save_dict
                with open(os.path.join(gen_save_path, "population.pickle"), "xb") as f:
                    pickle.dump(pop, f)
                with open(os.path.join(gen_save_path, "fitness.pickle"), "xb") as f:
                    pickle.dump(fitness, f)
                with open(os.path.join(gen_save_path, "optimizer.json"), "x", encoding="utf8") as f:
                    f.write(str(keras.utils.serialize_keras_object(optimizer)))
                with open(os.path.join(gen_save_path, "loss.json"), "x", encoding="utf8") as f:
                    f.write(str(keras.utils.serialize_keras_object(loss_metric)))
                with open(os.path.join(gen_save_path, "histories.pickle"), "xb") as f:
                    pickle.dump(histories, f)
                with open(os.path.join(gen_save_path, "train_tensor.pickle"), "xb") as f:
                    pickle.dump(train_tensor, f)
                with open(os.path.join(gen_save_path, "train_target.pickle"), "xb") as f:
                    pickle.dump(train_target, f)
                with open(os.path.join(gen_save_path, "val_tensor.pickle"), "xb") as f:
                    pickle.dump(val_tensor, f)
                with open(os.path.join(gen_save_path, "val_target.pickle"), "xb") as f:
                    pickle.dump(val_target, f)
            if gen_num < (generations - 1):
                # Reproduction
                if len(fitness) != len(pop):
                    raise UserWarning(
                        "fitness len is not the same as pop len! {} vs {}".format(
                            len(fitness), len(pop)
                        )
                    )

                keep_num = round(len(pop) * keep_fraction)
                reprod_num = round(len(pop) * reprod_fraction)
                children_per_indiv = round((pop_size - keep_num) / reprod_num)

                new_pop = [pop[ind] for ind in argsorted[:keep_num]]
                for reprod_index in argsorted[:reprod_num]:
                    reprod_indiv = pop[reprod_index]
                    children = reprod_indiv.gen_children(
                        clones=child_clones,
                        variants=children_per_indiv - child_clones,
                    )
                    new_pop.extend(children)
                pop = new_pop

        return pop, fitness


def main(argv, *args):
    x = pd.DataFrame([[i] * 10 for i in range(100)])
    y = pd.Series([99 - i for i in range(100)])
    print(next(train_val_tensor_iter(200, x, y, preprocess=False)))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
