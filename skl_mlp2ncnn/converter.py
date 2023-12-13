import argparse

import joblib

from skl_mlp2ncnn import convert


def _parse_args():
    parser = argparse.ArgumentParser()

    # sklearn MLP
    parser.add_argument(
        "-s",
        "--skl-mlp",
        type=str,
        help="path scikit-learn MLP model",
        required=True,
    )

    # ncnn
    parser.add_argument(
        "-n",
        "--ncnn",
        type=str,
        help="prefix of converted ncnn model",
        default="ncnn",
    )

    # class file
    parser.add_argument(
        "-c",
        "--class-table",
        type=str,
        help="path to class table file for MLPClassifier",
        default="class_table.txt",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # parse arg
    args = _parse_args()

    # convert
    convert(
        skl_mlp=joblib.load(args.skl_mlp),
        ncnn=args.ncnn,
        class_table=args.class_table,
        verbose=True
    )
