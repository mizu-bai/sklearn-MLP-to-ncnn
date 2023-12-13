# sklearn MLP to ncnn

A sklearn MLP to ncnn converter.

## Installation

```
$ git clone https://github.com/mizu-bai/sklearn-MLP-to-ncnn.git
$ pip install sklearn-MLP-to-ncnn
```

## Usage

```python3
from skl_mlp2ncnn import convert

# for MLPClassifier
convert(
    skl_mlp=classifier,
    ncnn="classifier",
    class_table="class_table.txt",
    verbose=True,
)

# for MLPRegressor
convert(
    skl_mlp=regressor,
    ncnn="regressor",
    verbose=True,
)
```

Examples for `MLPClassifier` and `MLPRegressor` are avaliable in directory `example/`.
