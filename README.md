# sklearn MLP to ncnn

A sklearn MLP to ncnn converter.

## Installation

```
$ git clone https://github.com/mizu-bai/sklearn-MLP-to-ncnn.git
$ pip install sklearn-MLP-to-ncnn
```

## Usage (CLI)

### Help

```bash
$ python3 -m skl_mlp2ncnn.converter -h
usage: converter.py [-h] -s SKL_MLP [-n NCNN] [-c CLASS_TABLE]

optional arguments:
-h, --help            show this help message and exit
-s SKL_MLP, --skl-mlp SKL_MLP
                        path to scikit-learn MLP model
-n NCNN, --ncnn NCNN  prefix of converted ncnn model
-c CLASS_TABLE, --class-table CLASS_TABLE
                        path to class table file for MLPClassifier
```

### `MLPClassifier`

```bash
$ python3 -m skl_mlp2ncnn.converter -s mnist_classifier.pkl -n mnist_classifier -c mnist.txt
Input model is MLPClassfier model.
classes will be saved to file mnist.txt.
================================================================================
                                    INFO                                      
--------------------------------------------------------------------------------
    Layer 1: Input
        Input shape: (784,)
--------------------------------------------------------------------------------
    Layer 2: InnerProduct
        Output shape: (512,)
--------------------------------------------------------------------------------
    Layer 3: ReLU
--------------------------------------------------------------------------------
    Layer 4: InnerProduct
        Output shape: (512,)
--------------------------------------------------------------------------------
    Layer 5: ReLU
--------------------------------------------------------------------------------
    Layer 6: InnerProduct
        Output shape: (10,)
--------------------------------------------------------------------------------
    Layer 7: Softmax
--------------------------------------------------------------------------------
    Classes: ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9']
================================================================================
Done! Model converted to mnist_classifier.param and mnist_classifier.bin.
Class table saved to mnist.txt.
```

### `MLPRegressor`

```bash
$ python3 -m skl_mlp2ncnn.converter -s mnist_autoencoder.pkl -n mnist_autoencoder
Input model is MLPRegressor model.
================================================================================
                                    INFO                                      
--------------------------------------------------------------------------------
    Layer 1: Input
        Input shape: (784,)
--------------------------------------------------------------------------------
    Layer 2: InnerProduct
        Output shape: (64,)
--------------------------------------------------------------------------------
    Layer 3: ReLU
--------------------------------------------------------------------------------
    Layer 4: InnerProduct
        Output shape: (784,)
================================================================================
Done! Model converted to mnist_autoencoder.param and mnist_autoencoder.bin.
```

## Usage (API)

```python3
from skl_mlp2ncnn import convert

# for MLPClassifier
convert(
    skl_mlp=classifier,
    ncnn="classifier",
    class_table="class_table.txt",
    verbose=True,  # print info
)

# for MLPRegressor
convert(
    skl_mlp=regressor,
    ncnn="regressor",
    verbose=True,  # print info
)
```

## Examples

Here are notebooks about training and converting scikit-learn MLP models:

- [`MLPClassifier`](https://github.com/mizu-bai/sklearn-MLP-to-ncnn/tree/main/example/MLPClassifier)
- [`MLPRegressor`](https://github.com/mizu-bai/sklearn-MLP-to-ncnn/tree/main/example/MLPRegressor)
