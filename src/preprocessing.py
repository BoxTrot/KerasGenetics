import json
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn import linear_model as lm

from string_convert import str_convert


def preprocess_csv_data(
        path="Data/data pruned full with covariates.csv",
        save_dir="Data",
        train_amount=0.6,
        val_amount=0.2,
        transform_cat=True,
        transform_num=True,
        prefix="preprocessed",
):
    if (train_amount < 0) or (val_amount < 0) or ((train_amount + val_amount) > 1):
        raise UserWarning("train_amount and val_amount must be >= 0 and =< 1!")
    df = pd.read_csv(path, index_col=0)
    df = df.drop(
        columns=[
            "D_Birth",
            "Sex",
            "D_Recrui",
            "Cntr_A",
            "D_Bld_Coll",
            "Batch_Number",
            "Well_Position",
            "Whr_C",
            "Bmi_C",
            "BMI_WHO",
            "BATCH",
        ]
    )

    # Ensure column names are valid variable names
    col_mapping = {}
    for each in df.columns:
        col_mapping[each] = str_convert(str(each))
    df = df.rename(columns=col_mapping)

    num_cols = [
        str_convert(i)
        for i in [
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
            "Age_Blood",
            "Height_C",
            "Weight_C",
            "Hip_C",
            "Waist_C",
        ]
    ]
    cat_cols = [str_convert(i) for i in ["Fasting_C", "Smoke_Stat"]]

    if transform_cat:
        one_hot_encodings = {}
        # one-hot encode categorical columns
        for col in cat_cols:
            values = df.loc[:, col].unique()
            values.sort()
            encoding: dict[Any, list[int]] = {}

            for i in range(len(values)):
                encoding[int(values[i])] = [0] * len(values)
                encoding[int(values[i])][i] = 1

            encoded_col_names = [col + "_{}".format(int(i)) for i in values]
            encoded_col_df = pd.DataFrame(index=df.index, columns=encoded_col_names)

            for ind in df.index:
                v = df.at[ind, col]
                if v in encoding:
                    encoded_col_df.loc[ind] = encoding[v]
            df = df.drop(columns=col)
            df = pd.concat([df, encoded_col_df], axis=1)
            one_hot_encodings[col] = encoding
        with open(
                os.path.join(save_dir, "{}_cat_encoding.json".format(prefix)), "w"
        ) as f:
            json.dump(one_hot_encodings, f, indent=4)

    # shuffle df with preset seed to maintain order. number from random.org
    df = df.sample(frac=1, random_state=228034).reset_index(drop=True)
    index_1 = int(train_amount * df.shape[0])
    if val_amount >= 0.0:
        index_2 = int((train_amount + val_amount) * df.shape[0])
        df_train: pd.DataFrame = df.iloc[:index_1].copy()
        df_validation: pd.DataFrame = df.iloc[index_1:index_2].copy()
        df_testing: pd.DataFrame = df.iloc[index_2:].copy()
    else:
        df_train: pd.DataFrame = df.iloc[:index_1].copy()
        df_validation: pd.DataFrame = pd.DataFrame()
        df_testing: pd.DataFrame = df.iloc[index_1:].copy()

    df_train_stats = df_train.loc[:, num_cols].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    df_train_stats.to_csv(
        os.path.join(save_dir, "{}_train_stat_before.csv".format(prefix))
    )

    if transform_num:
        # log-transform numerical columns
        df_train.loc[:, num_cols] = np.log10(df_train.loc[:, num_cols])
        if not df_validation.empty:
            df_validation.loc[:, num_cols] = np.log10(df_validation.loc[:, num_cols])
        df_testing.loc[:, num_cols] = np.log10(df_testing.loc[:, num_cols])
        # normalise numerical columns
        for col in num_cols:
            col_mean = df_train_stats.at["mean", col]
            col_std = df_train_stats.at["std", col]

            train_col = df_train.loc[:, col].copy()
            df_train.loc[:, col] = (train_col - col_mean) / col_std

            if not df_validation.empty:
                val_col = df_validation.loc[:, col].copy()
                df_validation.loc[:, col] = (val_col - col_mean) / col_std

            test_col = df_testing.loc[:, col].copy()
            df_testing.loc[:, col] = (test_col - col_mean) / col_std

        df_train_stats_new = df_train.loc[:, num_cols].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        df_train_stats_new.to_csv(
            os.path.join(save_dir, "{}_train_stat_after.csv".format(prefix))
        )

    df_train.to_csv(os.path.join(save_dir, "{}_train.csv".format(prefix)))
    if not df_validation.empty:
        df_validation.to_csv(os.path.join(save_dir, "{}_validation.csv".format(prefix)))
    df_testing.to_csv(os.path.join(save_dir, "{}_testing.csv".format(prefix)))


def transform_data(
        train: pd.DataFrame,
        val: pd.DataFrame = None,
        test: pd.DataFrame = None,
        transform_cat=True,
        transform_num=True,
        log_transform=True,
) -> tuple[pd.DataFrame, ...]:
    _train = train.copy()
    if val is not None:
        _val = val.copy()
    else:
        _val = None
    if test is not None:
        _test = test.copy()
    else:
        _test = None
    _df = pd.concat([train, val, test])

    try:
        _train.drop(
            columns=[
                "D_Birth",
                "Sex",
                "D_Recrui",
                "Cntr_A",
                "D_Bld_Coll",
                "Batch_Number",
                "Well_Position",
                "Whr_C",
                "Bmi_C",
                "BMI_WHO",
                "BATCH",
            ]
        )
    except KeyError:
        pass
    if _val is not None:
        try:
            _val.drop(
                columns=[
                    "D_Birth",
                    "Sex",
                    "D_Recrui",
                    "Cntr_A",
                    "D_Bld_Coll",
                    "Batch_Number",
                    "Well_Position",
                    "Whr_C",
                    "Bmi_C",
                    "BMI_WHO",
                    "BATCH",
                ]
            )
        except KeyError:
            pass
    if _test is not None:
        try:
            _test.drop(
                columns=[
                    "D_Birth",
                    "Sex",
                    "D_Recrui",
                    "Cntr_A",
                    "D_Bld_Coll",
                    "Batch_Number",
                    "Well_Position",
                    "Whr_C",
                    "Bmi_C",
                    "BMI_WHO",
                    "BATCH",
                ]
            )
        except KeyError:
            pass

    numerical_cols = [
        str_convert(i)
        for i in [
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
            "Age_Blood",
            "Height_C",
            "Weight_C",
            "Hip_C",
            "Waist_C",
        ]
    ]
    categorical_cols = [str_convert(i) for i in ["Fasting_C", "Smoke_Stat"]]
    for label in deepcopy(numerical_cols):
        if not (label in _df.columns):
            numerical_cols.remove(label)
    for label in deepcopy(categorical_cols):
        if not (label in _df.columns):
            categorical_cols.remove(label)
    if transform_cat and len(categorical_cols) > 0:
        one_hot_encodings = {}
        # one-hot encode categorical columns

        for col in categorical_cols:
            values = _df.loc[:, col].unique()
            values.sort()
            encoding: dict[Any, list[int]] = {}

            for i in range(len(values)):
                encoding[int(values[i])] = [0] * len(values)
                encoding[int(values[i])][i] = 1

            encoded_col_names = [col + "_{}".format(int(i)) for i in values]

            encoded_col_train = pd.DataFrame(
                index=_train.index, columns=encoded_col_names
            )
            for ind in _train.index:
                v = _train.at[ind, col]
                if v in encoding:
                    encoded_col_train.loc[ind] = encoding[v]
            _train = _train.drop(columns=col)
            _train = pd.concat([_train, encoded_col_train], axis=1)
            one_hot_encodings[col] = encoding

            if _val is not None:
                encoded_col_val = pd.DataFrame(
                    index=_val.index, columns=encoded_col_names
                )
                for ind in _val.index:
                    v = _val.at[ind, col]
                    if v in encoding:
                        encoded_col_val.loc[ind] = encoding[v]
                _val = _val.drop(columns=col)
                _val = pd.concat([_val, encoded_col_val], axis=1)
                one_hot_encodings[col] = encoding

            if _test is not None:
                encoded_col_test = pd.DataFrame(
                    index=_test.index, columns=encoded_col_names
                )
                for ind in _test.index:
                    v = _test.at[ind, col]
                    if v in encoding:
                        encoded_col_test.loc[ind] = encoding[v]
                _test = _test.drop(columns=col)
                _test = pd.concat([_test, encoded_col_test], axis=1)
                one_hot_encodings[col] = encoding

    if log_transform and len(numerical_cols) > 0:
        for col in numerical_cols:
            train_col = _train.loc[:, col].copy()
            _train.loc[:, col] = train_col.apply(np.log)

            if _val is not None:
                val_col = _val.loc[:, col].copy()
                _val.loc[:, col] = val_col.apply(np.log)

            if _test is not None:
                test_col = _test.loc[:, col].copy()
                _test.loc[:, col] = test_col.apply(np.log)

    if transform_num and len(numerical_cols) > 0:
        df_train_stats = _train.loc[:, numerical_cols].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        # normalise numerical columns
        for col in numerical_cols:
            col_mean = df_train_stats.at["mean", col]
            col_std = df_train_stats.at["std", col]
            train_col = _train.loc[:, col].copy()
            _train.loc[:, col] = (train_col - col_mean) / col_std

            if _val is not None:
                val_col = _val.loc[:, col].copy()
                _val.loc[:, col] = (val_col - col_mean) / col_std

            if _test is not None:
                test_col = _test.loc[:, col].copy()
                _test.loc[:, col] = (test_col - col_mean) / col_std

    if _val is None and _test is None:
        return _train
    else:
        out = [_train]
        if _val is not None:
            out.append(_val)
        if _test is not None:
            out.append(_test)
        return tuple(out)


def control_confounds(
        train_path="Data/preprocessed_train.csv",
        val_path="Data/preprocessed_validation.csv",
        test_path="Data/preprocessed_testing.csv",
        save_dir="Data",
):
    feature_cols = [
        str_convert(i)
        for i in [
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
        ]
    ]

    # confounds_num are numerical confounds, confounds_cat are categorical confounds
    confounds_num = [
        str_convert(i)
        for i in ["Age_Blood", "Height_C", "Weight_C", "Hip_C", "Waist_C"]
    ]
    confounds_cat = [
        str_convert(i)
        for i in [
            "Fasting_C_0",
            "Fasting_C_1",
            "Fasting_C_2",
            "Smoke_Stat_1",
            "Smoke_Stat_2",
            "Smoke_Stat_3",
            "Smoke_Stat_4",
        ]
    ]

    train = pd.read_csv(train_path, index_col=0)
    val = pd.read_csv(val_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    new_cols = ["sample_identification", "case_control"] + feature_cols
    train_new = train.copy().drop(columns=confounds_num + confounds_cat)
    val_new = val.copy().drop(columns=confounds_num + confounds_cat)
    test_new = test.copy().drop(columns=confounds_num + confounds_cat)

    stats_types = ["mean", "std", "min", "5%", "25%", "50%", "75%", "95%", "max"]
    output_stats = pd.DataFrame(columns=feature_cols, index=stats_types)

    for feat_col in feature_cols:
        linreg = lm.LinearRegression()
        linreg.fit(train.loc[:, confounds_num + confounds_cat], train.loc[:, feat_col])
        adj_col: pd.Series = train.loc[:, feat_col].copy()
        adj_col -= linreg.predict(train.loc[:, confounds_num + confounds_cat])
        adj_col_mean = adj_col.mean()
        adj_col_std = adj_col.std()
        train_new.loc[:, feat_col] = (adj_col - adj_col_mean) / adj_col_std
        numcat_stats = train_new.loc[:, feat_col].describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        output_stats.loc[:, feat_col] = numcat_stats

        val_adj_col: pd.Series = val.loc[:, feat_col].copy()
        val_adj_col -= linreg.predict(val.loc[:, confounds_num + confounds_cat])
        val_new.loc[:, feat_col] = (val_adj_col - adj_col_mean) / adj_col_std

        test_adj_col: pd.Series = test.loc[:, feat_col].copy()
        test_adj_col -= linreg.predict(test.loc[:, confounds_num + confounds_cat])
        test_new.loc[:, feat_col] = (test_adj_col - adj_col_mean) / adj_col_std

    print(output_stats)
    output_stats.to_csv(os.path.join(save_dir, "adjust_stats.csv"))
    train_new.to_csv(os.path.join(save_dir, "adjust_train.csv"))
    val_new.to_csv(os.path.join(save_dir, "adjust_validation.csv"))
    test_new.to_csv(os.path.join(save_dir, "adjust_testing.csv"))


def load_data(
        data_dir: str,
        files: list[str],
        columns_to_remove: list = None,
        indices_to_remove: list = None,
        index_col=None,
):
    if indices_to_remove is None:
        indices_to_remove = []
    if columns_to_remove is None:
        columns_to_remove = []

    output = []

    for file in files:
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath, index_col=index_col)
        cols = deepcopy(columns_to_remove)
        rows = deepcopy(indices_to_remove)
        for c in cols:
            if c in df.columns:
                df = df.drop(columns=c)
        for r in rows:
            if r in df.index:
                df = df.drop(index=r)
        output.append(df)

    return output
