#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def lin_relax(items, capacity):
	choc = [0]*len(items)
	for i in range(len(choc)):
		choc[i] = items[i].value/items[i].weight
	choc.sort()
	rem_cap = capacity
	zaebest = 0
	i = len(items) - 1
	while rem_cap > 0 or i >= 0:
		tt = min(items[i].weight, rem_cap)
		zaebest += choc[i]*tt
		rem_cap -= tt
		i -= 1
	return zaebest
def solve_it(input_data):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = input_data.split('\n')

	firstLine = lines[0].split()
	item_count = int(firstLine[0])
	capacity = int(firstLine[1])

	items = []

	for i in range(1, item_count+1):
		line = lines[i]
		parts = line.split()
		items.append(Item(i-1, int(parts[0]), int(parts[1])))
	#print(items[10].weight)

	value = 0

	
	weight = 0
	taken = [0]*len(items)

	#dynamic
	
	#table_building

	table = [[0 for x in range(capacity+1)] for x in range(item_count+1)]
	for i in range(0, item_count+1):
		for j in range(0, capacity+1):
			if i==0 or j==0:
				table[i][j] = 0
			elif items[i-1].weight <= j:
				table[i][j] = max(items[i-1].value + table[i-1][j-items[i-1].weight], table[i-1][j]) 
			else:
				table[i][j] = table[i-1][j]
	maxim = table[item_count][capacity]
	#print(table)
	value = maxim
	#backtracking
	cap = capacity
	for i in range(item_count, 0, -1):
		if table[i][cap] != table[i-1][cap]:
			cap -= items[i-1].weight
			taken[i-1] = 1


	
	# prepare the solution in the specified output format
	output_data = str(value) + ' ' + str(1) + '\n'
	output_data += ' '.join(map(str, taken))
	return output_data


if __name__ == '__main__':
	import sys
	if len(sys.argv) > 1:
		file_location = sys.argv[1].strip()
		with open(file_location, 'r') as input_data_file:
			input_data = input_data_file.read()
		print(solve_it(input_data))
	else:
		print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

