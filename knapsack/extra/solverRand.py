#!/usr/bin/python
# -*- coding: utf-8 -*-
from random import randrange
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def lin_relax(itemss, capacity):
	choc = [0]*len(itemss)
	for i in range(len(choc)):
		choc[i] = [itemss[i].value/itemss[i].weight, itemss[i].weight]
	choc.sort()
	rem_cap = capacity
	zaebest = 0
	i = len(itemss) - 1
	while rem_cap > 0 and i >= 0:
		tt = min(choc[i][1], rem_cap)
		zaebest += choc[i][0]*tt
		rem_cap -= tt
		i -= 1
	return zaebest
def dynamic(items, capacity):
	#table_building
	value = 0
	item_count = len(items)
	weight = 0
	taken = [0]*len(items)
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
	table.clear()
	taken.append(value)

	return taken


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


	c = 0
	cho = [0]*len(items)

	for i in range(len(cho)):
		cho[i] = [items[i].weight,items[i].value, items[i].index]
	cho.sort()
	#print(cho)
	rem_cap = capacity
	stop = 0
	while rem_cap > cho[0][0] and c <= item_count and stop < capacity:
		i = randrange(len(cho))
		if items[i].weight <= rem_cap and taken[i] == 0:
			value += items[i].value
			taken[i] = 1
			cho.remove([items[i].weight,items[i].value, items[i].index])
			c += 1
			rem_cap -= items[i].weight
			#print ("1)", i)
		elif taken[i] == 0:
			tempi = 0
			while rem_cap >= cho[tempi][0]:
				tempi += 1

			i = randrange(tempi)
			value += cho[i][1]
			taken[cho[i][2]] = 1
			rem_cap -= cho[i][0]
			cho.pop(i)

			c += 1
			#print ("2)", cho[i][2])
		else:
			#print("fuck")
			stop += 1
		#print(value)
		

	s = 0
#	for i in range(len(taken)):
#		s += taken[i]*items[i].value
	#print("sum", s)
	#print(value)
	# prepare the solution in the specified output format
	output_data = str(value) + ' ' + str(0) + '\n'
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

