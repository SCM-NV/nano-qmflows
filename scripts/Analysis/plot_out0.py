#! /usr/bin/env python
import ast
import numpy as np
import matplotlib.pyplot as plt
from interactive import ask_question


def parse_file(f):
    return np.genfromtxt(f)[:, 3::2].transpose()


def return_file(f3):
    return str(np.genfromtxt(f3, max_rows=1, dtype='str', delimiter='notinthere'))


def make_plot(arr, labels, states):
    for i in states:
        plt.plot(arr[i], label=labels['states'][i][0])
    plt.xlabel('time [fs]')
    plt.legend()
    plt.show()


def main():
    inp = './out/out0'
    arr = parse_file(inp)
    inp2 = './pyx.out'
    x = ast.literal_eval(return_file(inp2))

    question = ('Which states do you want to plot? Provide ints \
    (first state: 0), space-separated. [Default: all] ')
    dim = len(arr)
    dfs = " ".join(str(i) for i in range(dim)).split()
    states = map(int, ask_question(question, default=dfs))
    make_plot(arr, x, states)


if __name__ == '__main__':
    main()
