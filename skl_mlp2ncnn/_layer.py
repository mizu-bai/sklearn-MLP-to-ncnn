from dataclasses import dataclass, field
from typing import List

import numpy as np

ACT_SKL_MLP_TO_NCNN = {
    "identity": None,
    "tanh": "TanH",
    "logistic": "Sigmoid",
    "relu": "ReLU",
    "softmax": "Softmax",
}


@dataclass
class Layer:
    layer_type: str
    layer_name: str
    input_count: int
    output_count: int
    input_blobs: List[str] = field(default_factory=list)
    output_blobs: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return \
            f"{self.layer_type:<20s}{self.layer_name:<20s}" \
            f" {self.input_count} {self.output_count}" \
            f" {' '.join(self.input_blobs)} {' '.join(self.output_blobs)}"


@dataclass
class InputLayer(Layer):
    """layer specific params"""
    w: int = 0  # param id: 0

    def __str__(self) -> str:
        return super().__str__() + f" 0={self.w}"


@dataclass
class InnerProductLayer(Layer):
    """layer specific params"""
    num_output: int = 0  # param id: 0
    bias_term: int = 0  # param id: 1
    weight_data_size: int = 0  # param id: 2

    """data"""
    weight_data: np.ndarray = np.array([], dtype=np.float32)
    bias_data: np.ndarray = np.array([], dtype=np.float32)

    def __str__(self) -> str:
        return super().__str__() + \
            f" 0={self.num_output} 1={self.bias_term}" \
            f" 2={self.weight_data_size}"


@dataclass
class TanHLayer(Layer):
    """layer specific params"""
    pass


@dataclass
class SigmoidLayer(Layer):
    """layer specific params"""
    pass


@dataclass
class ReLULayer(Layer):
    """layer specific params"""
    pass


@dataclass
class SoftmaxLayer(Layer):
    """layer specific params"""
    pass
