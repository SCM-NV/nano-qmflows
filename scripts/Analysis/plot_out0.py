#! /usr/bin/env python
import ast
import numpy as np
import matplotlib.pyplot as plt

def parse_file(f):
	return np.genfromtxt(f)[:,3::2].transpose()

def return_file(f3):
	return str(np.genfromtxt(f3, max_rows=1, dtype='str', delimiter='notinthere'))

def make_plot(arr,labels):
	plts = [plt.plot(arr[i],label=labels['states'][i][0]) for i in range(len(arr))]
	plt.xlabel('time [fs]')
	plt.legend()
	plt.show()

def main():
	inp = './out/out0'
	arr = parse_file(inp)
	inp2 = './pyx.out'
	x = ast.literal_eval(return_file(inp2))

	make_plot(arr,x)

main()
