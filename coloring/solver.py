#!/usr/bin/python
# -*- coding: utf-8 -*-
from copy import deepcopy
from random import randrange
def sortot(mass):
	massiv = deepcopy(mass)
	massiv.sort(reverse = True)
	for i in range(len(massiv)):
		massiv[i].pop(0)
		massiv[i] = massiv[i][0]
	return massiv
def randam(graphic):
	graph = deepcopy(graphic)
	constraints = []
	solutionR = [-1]*node_count
	for i in range(0, node_count):
		constraints.append([])
	temp_xi = []
	for i in range(node_count):
		temp_xi.append([])
		temp_xi[i] = [len(graph[i]),i]
	temp_xi.sort(reverse = True)


	for i in range(node_count):
		temp_xi[i].pop(0)
		temp_xi[i] = temp_xi[i][0]
	current = 0
	while current <= node_count and solutionR.count(-1) > 0:
		temp_xi = []
		for i in range(node_count):
			temp_xi.append([])
			temp_xi[i] = [len(graph[i]),i]
		temp_xi.sort(reverse = True)
		for i in range(node_count):
			temp_xi[i].pop(0)
			temp_xi[i] = temp_xi[i][0]
		while len(temp_xi) > 0:
			l = randrange(len(temp_xi))
			i = temp_xi[l]
			if constraints[i].count(current) > 0:
				temp_xi.remove(i)
			elif solutionR[i] != -1:
				temp_xi.remove(i)
			else:
				solutionR[i] = current
				for j in graph[i]:
					constraints[j].append(current)
				temp_xi.remove(i)
		current += 1
	solutionR.append(current)
	return solutionR


def solve_it(input_data):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = input_data.split('\n')

	first_line = lines[0].split()
	global node_count
	node_count = int(first_line[0])
	edge_count = int(first_line[1])


	edges = []
	for i in range(1, edge_count + 1):
		line = lines[i]
		parts = line.split()
		edges.append((int(parts[0]), int(parts[1])))

	# build a trivial solution
	# every node has its own color
	solution = [-1]*node_count
	solution1 = [-1]*node_count
	solution2 = [-1]*node_count
	solution3 = [-1]*node_count
	print ("nodes", node_count)
	print ("edges", edge_count)


	graph = []
	constraints = []
	does_connected = []
	#print (edges)
	#print (graph)
	#print (edges[1])
	#print (edges[1][0])
	for i in range(0, node_count):
		graph.append([])
		constraints.append([])
		does_connected.append([0, i])

		
		for j in range(0, edge_count):
			if edges[j][0] == i:
				graph[i].append(edges[j][1])
			elif edges[j][1] == i:
				graph[i].append(edges[j][0])
	#does_connected[3][0] = -1
	temp_xi = []
	for i in range(node_count):
		temp_xi.append([])
		temp_xi[i] = [len(graph[i]),i]
	temp_xi.sort(reverse = True)

	#print (does_connected)
	#print (temp_xi)
	for i in range(node_count):
		temp_xi[i].pop(0)
		temp_xi[i] = temp_xi[i][0]
	#print (temp_xi)
	#print (solution)
	current = 0
	while current <= node_count and solution1.count(-1) > 0:
		for i in range(len(does_connected)):
			does_connected[i][0] = 0
		for i in temp_xi:
			if solution1[i] == -1:
				solution1[i] = current
				for j in graph[i]:
					constraints[j].append(current)
					for k in graph[j]:
						does_connected[k][0] += 1 
				break
		#print (does_connected)
		temp_tyan = sortot(does_connected)
		#print (does_connected)
		for i in temp_tyan:
			if constraints[i].count(current) == 0 and solution1[i] == -1:
				solution1[i] = current
				for j in graph[i]:
					constraints[j].append(current)
					for k in graph[j]:
						does_connected[k][0] += 1 
				temp_tyan = sortot(does_connected)
		current += 1
	cols1 = current




	constraints = []
	for i in range(0, node_count):

		constraints.append([])

	current = 0
	while current <= node_count and solution2.count(-1) > 0:
		while does_connected.count(1) > 0:
			does_connected[does_connected.index(1)] = 0
		for i in temp_xi:
			if solution2[i] == -1:
				solution2[i] = current
				for j in graph[i]:
					constraints[j].append(current)
					for k in graph[j]:
						does_connected[k] = 1 
				break
		for i in temp_xi:
			if constraints[i].count(current) == 0 and solution2[i] == -1 and does_connected[i] == 1:
				solution2[i] = current
				for j in graph[i]:
					constraints[j].append(current)
					for k in graph[j]:
						does_connected[k] = 1 
		for i in temp_xi:
			if constraints[i].count(current) == 0 and solution2[i] == -1 and does_connected[i] == 0:
				solution2[i] = current
				for j in graph[i]:
					constraints[j].append(current)
		current += 1
	cols2 = current

	cols3 = node_count
	if edge_count < 20000:
		for i in range(5000):
			c = randam(graph)
			cc = c.pop()
			print (cc)
			if cols3 > cc:
				solution3 = deepcopy(c)
				cols3 = cc

	if cols2 <= cols1 and cols2 <= cols3:
		solution = solution2
		cols = cols2
		print ("2")
	elif cols1 <= cols2 and cols1 <= cols3:
		solution = solution1
		cols = cols1
		print ("1")
	elif cols3 <= cols2 and cols3 <= cols1:
		solution = solution3
		cols = cols3
		print ("3")

	# prepare the solution in the specified output format
	output_data = str(cols) + ' ' + str(0) + '\n'
	output_data += ' '.join(map(str, solution))

	return output_data


import sys

if __name__ == '__main__':
	import sys
	if len(sys.argv) > 1:
		file_location = sys.argv[1].strip()
		with open(file_location, 'r') as input_data_file:
			input_data = input_data_file.read()
		print(solve_it(input_data))
	else:
		print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

