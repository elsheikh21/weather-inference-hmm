import numpy as np


def write_to_file(output, file_name):
    '''
    Saves output to a text file.

    Arguments:
            output
            file_name
    '''
    with open(file_name, "w") as text_file:
        for entry in output:
            print(entry, file=text_file)


def dict_to_matrix(d):
    '''
    Takes dictionary and returns a matrix.
    '''

    lst = []

    for k, v in d.items():
        lst.append(v)

    mat = np.array(lst)
    return mat
