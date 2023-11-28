from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import math
from typing import Union, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from q_funcs import series_inner_product


class Classifier:
    def __init__(self, initial_dimensions: int):
        self.log_dim = int(math.ceil(np.log2(initial_dimensions)))
        self.dimensions = 2**self.log_dim
        self.train_vecs = {}
    def add_train_data(self, cl: str, vec: Union[list, np.ndarray]):
        vec = list(vec) + [0] * (self.dimensions - len(vec))
        if cl in self.train_vecs:
            self.train_vecs[cl] = np.add(vec, self.train_vecs[cl])
        else:
            self.train_vecs[cl] = vec
        return self.train_vecs

    def classify(self, test_vec: Union[list, np.ndarray], show_circuit: bool = False, print_counts: bool = False)-> str:
        test_vec = test_vec + [0]*(self.dimensions - len(test_vec))
        max_inner = None
        shown = False
        for cl, vec in self.train_vecs.items():
            if print_counts:
                print(f"Testing {test_vec} with {cl}, {vec}")
            if not shown:
                inner_product = series_inner_product(v_1=list(vec), v_2=test_vec, show_circuit=show_circuit,
                                                     print_counts=print_counts)
                shown = True
            else:
                inner_product = series_inner_product(v_1=list(vec), v_2=test_vec, print_counts=print_counts)
            if max_inner is None:
                max_inner = [cl, inner_product]
            elif inner_product > max_inner[1]:
                max_inner = [cl, inner_product]


        return max_inner[0]


if __name__ == '__main__':
    test = Classifier(3)
    test.add_train_data("one", [1,2,1,4])
    test.add_train_data('two', [1,6,3,3])
    test.add_train_data('three', [4,7,9,2])

    print(test.classify(np.array([1,2,1,4])))