import sys

"""
The class that will coordinate population initialisation, training, culling, and
reproduction.
Use asexual reproduction with mutation
"""


class Stager:
    def __init__(
            self,
            init_pop_generator: callable,
            loss_metric,

    ):
        pass

    def run(
            self,
            population: int,
            generations: int,
            loss_metric=None,
            trials_per_generation: int = 1,
            cull_fraction: float = 0.5,

    ):
        pass


def main(argv, *args):
    pass


if __name__ == '__main__':
    sys.exit(main(sys.argv))
