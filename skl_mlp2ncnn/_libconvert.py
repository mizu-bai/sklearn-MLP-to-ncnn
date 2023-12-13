from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from skl_mlp2ncnn._layer import (ACT_SKL_MLP_TO_NCNN, InnerProductLayer,
                                 InputLayer, Layer, ReLULayer, SigmoidLayer,
                                 SoftmaxLayer, TanHLayer)


def _parse_layers(
    skl_mlp: Union[MLPClassifier, MLPRegressor],
) -> List[Layer]:
    layer_list = []

    # input layer
    input_layer = InputLayer(
        layer_type="Input",
        layer_name="input",
        input_count=0,
        output_count=1,
        input_blobs=[],
        output_blobs=["input_blob"],
        w=skl_mlp.n_features_in_,
    )

    layer_list.append(input_layer)

    # hidden and output layers
    for idx in range(skl_mlp.n_layers_ - 1):
        # inner product
        inner_product_layer = InnerProductLayer(
            layer_type="InnerProduct",
            layer_name=f"layer_{idx}",
            input_count=1,
            output_count=1,
            input_blobs=layer_list[-1].output_blobs,
            output_blobs=[f"layer_{idx}_blob"],
            num_output=(skl_mlp.hidden_layer_sizes[idx] if idx < (
                skl_mlp.n_layers_ - 2) else skl_mlp.n_outputs_),
            bias_term=1,
            weight_data_size=skl_mlp.coefs_[idx].size,
            weight_data=skl_mlp.coefs_[idx],
            bias_data=skl_mlp.intercepts_[idx],
        )

        layer_list.append(inner_product_layer)

        # activation
        act_type = ACT_SKL_MLP_TO_NCNN[skl_mlp.activation if idx < (
            skl_mlp.n_layers_ - 2) else skl_mlp.out_activation_]

        if not act_type:
            continue

        act_layer = eval(f"{act_type}Layer")(
            layer_type=act_type,
            layer_name=f"{layer_list[-1].layer_name}_{act_type}".lower(),
            input_count=1,
            output_count=1,
            input_blobs=layer_list[-1].output_blobs,
            output_blobs=[
                f"{layer_list[-1].layer_name}_{act_type}_blob".lower()
            ],
        )

        layer_list.append(act_layer)

    return layer_list


def _print_layers_info(
    layer_list: List[Layer],
) -> None:
    print("=" * 80)
    print(f"{'INFO':^80s}")
    print("-" * 80)

    for (idx, layer) in enumerate(layer_list):
        print(f"    Layer {idx + 1}: {layer.layer_type}")

        if isinstance(layer, InputLayer):
            print(f"        Input shape: ({layer.w},)")
        elif isinstance(layer, InnerProductLayer):
            print(f"        Output shape: ({layer.num_output},)")

        if idx < len(layer_list) - 1:
            print("-" * 80)


def _write_param(
    layer_list: List[Layer],
    ncnn: str,
) -> None:
    with open(f"{ncnn}.param", "w") as f:
        # magic number
        f.write("7767517\n")

        # layer count and blob count
        f.write(f"{len(layer_list)} {len(layer_list)}\n")

        # write layers
        for layer in layer_list:
            f.write(f"{str(layer)}\n")


def _write_bin(
    layer_list: List[Layer],
    ncnn: str,
) -> None:
    with open(f"{ncnn}.bin", "wb") as f:
        for layer in layer_list:
            if hasattr(layer, "weight_data") and hasattr(layer, "bias_data"):
                # flag
                f.write(np.array(0, dtype=np.int32).tobytes())

                # weight
                f.write(layer.weight_data.T.astype(np.float32).tobytes())

                # bias
                if layer.bias_term == 1:
                    f.write(layer.bias_data.astype(np.float32).tobytes())


def convert(
    skl_mlp: Union[MLPClassifier, MLPRegressor],
    ncnn: Optional[str] = "ncnn",
    class_table: Optional[str] = "class_table.txt",
    verbose=False,
) -> Tuple[str, str]:
    """Convert scikit-learn MLP model to ncnn.

    Arguments:
        skl_mlp (Union[MLPClassifier, MLPRegressor]): scikit-learn MLP model.
        ncnn (Optional[str]): Path to converted ncnn model.
        class_table (Optional[str]): Path to class table. Only use with
            MLPClassifier model.

    Returns:
        param (str): Path to converted ncnn param file.
        bin (str): Path to converted ncnn bin file.
    """
    # check input type
    if isinstance(skl_mlp, MLPClassifier):
        print("Input model is MLPClassfier model.")
        print(f"classes will be saved to file {class_table}.")
    elif isinstance(skl_mlp, MLPRegressor):
        print("Input model is MLPRegressor model.")
    else:
        raise TypeError(f"Input model is {type(skl_mlp)}, not supported yet.")

    # parse layers
    layer_list = _parse_layers(skl_mlp)

    # print layers info if required
    if verbose:
        _print_layers_info(layer_list)

        if isinstance(skl_mlp, MLPClassifier):
            print("-" * 80)
            print(f"    Classes: {skl_mlp.classes_}")

        print("=" * 80)

    # write param file
    _write_param(layer_list, ncnn)

    # write bin file
    _write_bin(layer_list, ncnn)

    print(f"Done! Model converted to {ncnn}.param and {ncnn}.bin.")

    if isinstance(skl_mlp, MLPClassifier):
        np.savetxt(class_table, skl_mlp.classes_, fmt="%s")
        print(f"Class table saved to {class_table}.")

    return f"{ncnn}.param", f"{ncnn}.bin"
