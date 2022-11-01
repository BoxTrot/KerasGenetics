import sys

import pandas as pd

import stager
from string_convert import str_convert


def main(argv, *args):

    features = list(
        map(
            str_convert,
            [
                "His",
                "Ile",
                "Leu",
                "Lys",
                "Met",
                "Phe",
                "Ser",
                "Thr",
                "Trp",
                "Val",
                "PC ae C34:3",
                "lysoPC a C18:2",
                "SM C18:0",
            ],
        )
    )
    target = "case_control"
    train_val = pd.read_csv("../data/split_train.csv", index_col=0).loc[:, [target] + features]
    train_val_target = train_val.pop(target)

    stage = stager.Stager()
    stage.run(
        train_val= train_val,
        train_val_target = train_val_target,
        child_clones=0,
        verbose=3
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
