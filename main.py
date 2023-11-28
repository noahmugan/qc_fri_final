#The main file which runs the user interface. To use the scripts, run this.

import pickle
import numpy as np
from classifier import Classifier
from q_funcs import series_inner_product


def get_pickled_data(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as file:
        vector = pickle.load(file)
    return vector

def pickle_data(file_path: str = "/Users/noahmugan/Dropbox/fall_fest_23/") -> None:
    with open(file_path, 'wb') as file:
        # Write the NumPy array to the pickle file
        pickle.dump(file_path, file)

class SizeError(Exception):
    pass

def blank_terminal():
    """
    Helper to blank the terminal.
    """
    print("\033c")

def arbitrary():
    """
    Allows for the user to classify a vector with arbitrary classes
    :return:
    """
    blank_terminal()
    #First the user inputs training vectors and their classifications
    print("First, input training vectors as either a .pkl file or a list.")
    classifier = None
    input_type = input('Input "pkl" to load a pkl file or "lst" to load a list:\n').strip()
    while input_type != "done" or classifier is None:
        vector = None
        if input_type == 'pkl':
            filepath = input("Input the filepath for the .pkl file:\n").strip()
            try:
                vector = get_pickled_data(filepath)
            except FileNotFoundError:
                print("Invalid file path")
        elif input_type == 'lst':
            vec_list = input(
                "\nInput the vector as a list of floats or integers separated by spaces.\n").strip().split()
            try:
                vector = [float(i) for i in vec_list]
            except ValueError:
                print(
                    'Invalid input. List values must be floats or integers')
        else:
            print("Invalid input.")
        if input_type in ("pkl", "lst") and vector is not None:
            try:
                if classifier is None:
                    classifier = Classifier(len(vector))
                    print()
                else:
                    if len(vector) > classifier.dimensions:
                        raise SizeError
                    print(f"\nYour current classifications are {[i for i in classifier.train_vecs.keys()]}")
                cl = input(
                    "Input the classification for the just-entered vector.\nThis can be more data for an existing classification or data for a new one:\n").strip()
                classifier.add_train_data(cl, vector)
            except SizeError:
                print(f"Error - Dimensions of new data do not match previously established dimensions: {classifier.dimensions}")

        if classifier is None:
            input_type = input('\nInput "pkl" to load a pkl file or "lst" to load a list\n').strip()
        else:
            input_type = input('\nInput "pkl" to load a pkl file, "lst" to load a list, or "done" to finish:\n').strip()
    #Once the training vectors are all in, the user can input a test vector
    valid = False
    while not valid:
        test_vector = input(
            "\nInput the test vector as a list of floats or integers separated by spaces.\n").strip().split()
        if len(test_vector) <= classifier.dimensions:
            try:
                test_vector = [float(i) for i in test_vector]
                valid = True
            except ValueError:
                print(
                    'Invalid input. List values must be floats or integers')
        else:
            print(f"Error - Dimensions of new data do not match previously established dimensions: {classifier.dimensions}")
    show_circuit = input("Would you like to see the resulting circuit? (y/n)\n").lower().strip() == 'y'
    print_counts = input("Would you like to see the measured counts? (y/n)\n").lower().strip() == 'y'
    match = classifier.classify(test_vector, show_circuit=show_circuit, print_counts=print_counts)
    print(f"\nBest match: {match}")
    go = input("\nContinue? (y/n)\n").lower().strip()
    return go == 'y'

def single_product():
    """
    Allows for the user to calculate the inner product between two vectors
    :return:
    """
    blank_terminal()
    vec_1 = None
    vec_2 = None
    #Input the first vector
    print("Input the first vector as either a .pkl file or a list.")
    while vec_1 is None:
        input_type = input('Input "pkl" to load a pkl file or "lst" to load a list:\n')
        if input_type == 'pkl':
            filepath = input("Input the filepath for the .pkl file:\n").strip()
            try:
                vec_1 = get_pickled_data(filepath)
            except FileNotFoundError:
                print("Invalid file path")
        elif input_type == 'lst':
            vec_list = input(
                "\nInput the vector as a list of floats or integers separated by spaces.\n").strip().split()
            try:
                vec_1 = [float(i) for i in vec_list]
            except ValueError:
                print(
                    'Invalid input. List values must be floats or integers')
        else:
            print("Invalid input.")
    #Input the second vector
    print("\nInput the second vector as either a .pkl file or a list.")
    while vec_2 is None:
        input_type = input('Input "pkl" to load a pkl file or "lst" to load a list:\n')
        if input_type == 'pkl':
            filepath = input("Input the filepath for the .pkl file:\n").strip()
            try:
                vec_2 = get_pickled_data(filepath)

            except FileNotFoundError:
                print("Invalid file path")
        elif input_type == 'lst':
            vec_list = input(
                "\nInput the vector as a list of floats or integers separated by spaces.\n").strip().split()
            try:
                vec_2 = [float(i) for i in vec_list]
            except ValueError:
                print(
                    'Invalid input. List values must be floats or integers')
        else:
            print("Invalid input.")
    show_circuit = input("Would you like to see the resulting circuit? (y/n)\n").lower().strip() == 'y'
    inner_product = series_inner_product(v_1=vec_1, v_2=vec_2, show_circuit=show_circuit)
    print(f"\nInner product: {inner_product}")
    go = input("\nContinue? (y/n)\n").lower().strip()
    return go == 'y'

#The main loop
if __name__ == '__main__':
    blank_terminal()
    print("Hello. Welcome to the Qiskit-based innper-product approximator.")
    print("\nThis program uses a quantum algorithm to calculate the inner product between two vectors in logarithmic time. This approach allows us to efficiently classify vectors.")
    print("You can either classify a vector using your own training set or simply determine the inner product between two vectors.")
    choice = input('Type "classify" to start the classifier, "product" to calculate an inner product, or "stop" to exit the program.\n').lower().strip()
    while choice != 'stop':
        print()
        if choice == 'classify':
            go = arbitrary()
            if not go:
                choice = 'stop'
            else:
                blank_terminal()
                choice = input(
                    'What next? Type "classify" to start the classifier, "product" to calculate an inner product, or "stop" to exit the program.\n').lower().strip()

        elif choice == 'product':
            go = single_product()
            if not go:
                choice = 'stop'
            else:
                blank_terminal()
                choice = input(
                    'What next? Type "classify" to start the classifier, "product" to calculate an inner product, or "stop" to exit the program.\n').lower().strip()

        elif choice != 'stop':
            print("\nInvalid input.")
            choice = input(
                'Type "classify" to start the classifier, "product" to calculate an inner product, or "stop" to exit the program.\n').lower().strip()

