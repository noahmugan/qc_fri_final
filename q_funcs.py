#Defines the functions which run the inner-product circuit

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Qubit, Clbit
import math
from typing import Union, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import os
from pathlib import Path
from qiskit.visualization import plot_histogram


def _inner_product(v_1: Union[List[Union[float, int, complex]], np.ndarray],
                   v_2: Union[List[Union[float, int, complex]], np.ndarray],
                   qc: QuantumCircuit,
                   qr_1: QuantumRegister,
                   qr_2: QuantumRegister,
                   q_out: Qubit,
                   c_out: Clbit,
                   num_qubits: Optional[int] = None
                   )-> None:
    """
    Given vectors and their spots in a quantum circuit, it instantiates them into qubit states and performs the inner product test
    :param v_1:
    :param v_2:
    :param qc:
    :param qr_1:
    :param qr_2:
    :param q_out:
    :param c_out:
    :param num_qubits:
    :return:
    """
    if num_qubits is None:
        qc.initialize(v_1, qr_1, normalize=True)
        qc.initialize(v_2, qr_2, normalize=True)
    else:
        qc.initialize(v_1, qr_1[:num_qubits], normalize=True)
        qc.initialize(v_2, qr_2[:num_qubits], normalize=True)
    qc.h(q_out)
    for i in range(len(qr_1)):
        qc.cswap(q_out, qr_1[i], qr_2[i])
    qc.h(q_out)
    qc.measure(q_out, c_out)

def parallel_inner_product(v_1: Union[List[Union[float, int]], np.ndarray],
                        v_2: Union[List[Union[float, int]], np.ndarray],
                        names: Optional[List[str]] = None,
                        show_circuit: Optional[bool] = False,
                        print_counts: Optional[bool] = False
                        )->float:
    """
    A function which was meant to run all three inner product tests in parallel. This does not scale well at all. Use the series version.
    :param v_1:
    :param v_2:
    :param names:
    :param show_circuit:
    :param print_counts:
    :return:
    """
    log_dim = int(math.ceil(np.log2(len(v_1))))
    filled_v_1 = v_1 + [0] * (2 ** log_dim - len(v_1))
    filled_v_2 = v_2 + [0] * (2 ** log_dim - len(v_2))

    q_out = QuantumRegister(3, name="Output")
    qr_1 = QuantumRegister(log_dim, name = ("State 1" if names is None else f"{names[0]}"))
    qr_2 = QuantumRegister(log_dim, name=("State 2" if names is None else f"{names[1]}"))
    first_exp_qr_1 = QuantumRegister(log_dim + 1, name = ("1st E.S. 1" if names is None else f"Exp {names[0]}"))
    even_exp_qr_2 = QuantumRegister(log_dim + 1, name = ("E.E.S 2" if names is None else f"E.E. {names[1]}"))
    second_exp_qr_1 = QuantumRegister(log_dim + 1,
                                     name=("2nd E.S. 1" if names is None else f"Exp {names[0]}"))
    odd_exp_qr_2 = QuantumRegister(log_dim + 1, name=(
        "O.E.S. 1" if names is None else f"O.E {names[1]}"))
    cr = ClassicalRegister(3, name = 'Results')
    qc = QuantumCircuit(q_out,
                        qr_1,
                        qr_2,
                        first_exp_qr_1,
                        even_exp_qr_2,
                        second_exp_qr_1,
                        odd_exp_qr_2,
                        cr
                        )
    exp_v_1 = []
    even_exp_v_2 = []
    odd_exp_v_2 = []
    for i in filled_v_1:
        exp_v_1 += [i, 0] if i > 0 else [0, -1 * i]
    for i in filled_v_2:
        even_exp_v_2 += [i, 0] if i > 0 else [0, -1 * i]
        odd_exp_v_2 += [0, i] if i > 0 else [-1 * i, 0]
    _inner_product(
        v_1=filled_v_1,
        v_2=filled_v_2,
        qc=qc,
        qr_1=qr_1,
        qr_2=qr_2,
        q_out=q_out[0],
        c_out=cr[0]
    )
    _inner_product(
        v_1=exp_v_1,
        v_2=even_exp_v_2,
        qc=qc,
        qr_1=first_exp_qr_1,
        qr_2=even_exp_qr_2,
        q_out=q_out[1],
        c_out=cr[1]
    )
    _inner_product(
        v_1=exp_v_1,
        v_2=odd_exp_v_2,
        qc=qc,
        qr_1=second_exp_qr_1,
        qr_2=odd_exp_qr_2,
        q_out=q_out[2],
        c_out=cr[2]
    )

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    data = job.result().get_counts(qc)

    abs_counts = 0
    pos_counts = 0
    neg_counts = 0

    for result, count in data.items():
        if result[0] == '0':
            neg_counts += count
        if result[1] == '0':
            pos_counts += count
        if result[2] == '0':
            abs_counts += count

    abs_counts = 500 if abs_counts < 500 else abs_counts

    abs_product = np.sqrt((abs_counts * 0.001 - 0.5) * 2)

    inner_product = abs_product if pos_counts > neg_counts else (-1 * abs_product)

    if print_counts:
        print(f"Pos counts: {pos_counts}, Neg counts: {neg_counts}, Abs counts: {abs_counts}")
        print(f"Inner product: {inner_product}")
    if show_circuit:
        qc.draw('mpl')
        plt.show()
    return inner_product

def series_inner_product(v_1: Union[List[Union[float, int]], np.ndarray],
                        v_2: Union[List[Union[float, int]], np.ndarray],
                        names: Optional[List[str]] = None,
                        show_circuit: Optional[bool] = False,
                        print_counts: Optional[bool] = False
                        )->float:
    """
    Given two vectors, determines the inner product between them
    :param v_1:
    :param v_2:
    :param names:
    :param show_circuit:
    :param print_counts:
    :return:
    """
    if len(v_1) == 1:
        v_1 += [0]
    if len(v_1) < len(v_2):
        v_1 += [0]*(len(v_2)-len(v_1))
    if len(v_2) < len(v_1):
        v_2 += [0]*(len(v_1)-len(v_2))
    log_dim = int(math.ceil(np.log2(len(v_1))))
    filled_v_1 = v_1 + [0]*(2**log_dim-len(v_1))
    filled_v_2 = v_2 + [0] * (2 ** log_dim - len(v_2))

    q_out = QuantumRegister(1, name="Output")
    qr_1 = QuantumRegister(log_dim+1, name = ("State 1" if names is None else f"{names[0]}"))
    qr_2 = QuantumRegister(log_dim+1, name=("State 2" if names is None else f"{names[1]}"))
    cr = ClassicalRegister(3, name = 'Results')
    qc = QuantumCircuit(q_out,
                        qr_1,
                        qr_2,
                        cr
                        )
    exp_v_1 = []
    even_exp_v_2 = []
    odd_exp_v_2 = []
    for i in filled_v_1:
        exp_v_1 += [i, 0] if i > 0 else [0, -1 * i]
    for i in filled_v_2:
        even_exp_v_2 += [i, 0] if i > 0 else [0, -1 * i]
        odd_exp_v_2 += [0, i] if i > 0 else [-1 * i, 0]
    #Normal inner product test
    _inner_product(
        v_1=filled_v_1,
        v_2=filled_v_2,
        qc=qc,
        qr_1=qr_1,
        qr_2=qr_2,
        q_out=q_out[0],
        c_out=cr[0],
        num_qubits=log_dim
    )
    #Even-expanded inner product test
    qc.x(q_out[0]).c_if(cr[0], 1)
    _inner_product(
        v_1=exp_v_1,
        v_2=even_exp_v_2,
        qc=qc,
        qr_1=qr_1,
        qr_2=qr_2,
        q_out=q_out[0],
        c_out=cr[1]
    )
    #Odd-expanded inner product test
    qc.x(q_out[0]).c_if(cr[1], 1)
    _inner_product(
        v_1=exp_v_1,
        v_2=odd_exp_v_2,
        qc=qc,
        qr_1=qr_1,
        qr_2=qr_2,
        q_out=q_out[0],
        c_out=cr[2]
    )


    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    data = job.result().get_counts(qc)

    abs_counts = 0
    pos_counts = 0
    neg_counts = 0

    for result, count in data.items():
        if result[0] == '0':
            neg_counts += count
        if result[1] == '0':
            pos_counts += count
        if result[2] == '0':
            abs_counts += count

    abs_counts = 500 if abs_counts < 500 else abs_counts

    abs_product = np.sqrt((abs_counts * 0.001 - 0.5) * 2)

    inner_product = abs_product if pos_counts > neg_counts else (-1 * abs_product)

    if print_counts:
        #plot_histogram(data)
        print(f"Pos counts: {pos_counts}, Neg counts: {neg_counts}, Abs counts: {abs_counts}")
        print(f"Inner product: {inner_product}")
    if show_circuit:
        qc.draw('mpl')
        plt.show()
        # if 'noahmugan' in str(Path.home()):
        #     plt.savefig('/Users/noahmugan/Dropbox/classes/Quantum_Computing_FRI/final_project/series_circuit.png')
    return inner_product

if __name__ == '__main__':
    # times_lst = []
    # actual_inner_products = []
    # inner_product_error = []
    # for size in range(1, 21):
    #     runtime = 0
    #     for i in range(2):
    #         v_1 = [random.randint(-10,10) for j in range(size)]
    #         while all([j == 0 for j in v_1]):
    #             v_1 = [random.randint(-10, 10) for j in range(size)]
    #         v_2 = [random.randint(-10, 10) for j in range(size)]
    #         while all([j == 0 for j in v_2]):
    #             v_2 = [random.randint(-10, 10) for j in range(size)]
    #         mag_1 = math.sqrt(sum([i**2 for i in v_1]))
    #         v_1 = [i/mag_1 for i in v_1]
    #         mag_2 = math.sqrt(sum([i ** 2 for i in v_2]))
    #         v_2 = [i / mag_2 for i in v_2]
    #         print(f"Vector 1: {v_1}")
    #         print(f"Vector 2: {v_2}")
    #         real_product = sum([v_1[i] * v_2[i] for i in range(len(v_1))])
    #         actual_inner_products.append(real_product)
    #         print(f"Real product is: {real_product}")
    #         time_start = datetime.datetime.now()
    #         approx_product = series_inner_product(v_1, v_2, print_counts=True, show_circuit=False)
    #         time_end = datetime.datetime.now()
    #         time_diff = time_end-time_start
    #         runtime += time_diff.total_seconds()
    #         percent_error = abs((approx_product-real_product)/real_product)*100
    #         inner_product_error.append(percent_error)
    #     times_lst.append(runtime/5)
    #
    # plt.figure(1)
    # plt.plot(range(1, 21), times_lst, "o", color='red')
    # plt.xlabel("Vector Dimension")
    # plt.ylabel("Simulator Runtime (s)")
    # plt.title("Simulator Runtime vs Input Vector Dimension")
    # plt.show()
    #
    # plt.figure(2)
    # plt.plot(actual_inner_products, inner_product_error, "o", color='orange')
    # plt.xlabel("Expected Inner Product")
    # plt.ylabel("Approximation Error (%)")
    # plt.title("Percent error vs Expected Inner Product")
    # plt.show()

    v_1 = [1,6,-2,6,9,-2,5,8]
    v_2 = [5,-8,3,-5,0,-2,6,-8]
    mag_1 = math.sqrt(sum([i ** 2 for i in v_1]))
    v_1 = [i/mag_1 for i in v_1]
    mag_2 = math.sqrt(sum([i ** 2 for i in v_2]))
    v_2 = [i / mag_2 for i in v_2]
    print(f"Vector 1: {v_1}")
    print(f"Vector 2: {v_2}")
    real_product = sum([v_1[i] * v_2[i] for i in range(len(v_1))])
    print(f"Real product is: {real_product}")
    series_inner_product(v_1=v_1, v_2=v_2, show_circuit=True, print_counts=True)
