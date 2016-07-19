#! /usr/bin/env python
import ast
import numpy as np
def parse_file(f):
	arr = []
	first = True
	for line in f:
		line2 = line.split()
		i = 3
		while i<len(line2):
			if first:
				arr.append([])
			arr[int((i-3)/2)].append(float(line2[i]))
			i += 2
		first = False
	return arr

def return_file(f3):
	first = True
	for line in f3:
		if first:
			first = False
			return line
		else:
			first = False


inp = './out/out0'
arr = []
with open(inp) as f:
	arr = parse_file(f)
inp2 = './pyx.out'
x = {}
with open(inp2) as f2:
	x = ast.literal_eval(return_file(f2))

import matplotlib.pyplot as plt
i = 0
for j in arr:
	plt.plot(j,label=x['states'][i][0])
	i += 1

plt.legend()
plt.show()
