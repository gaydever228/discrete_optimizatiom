#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation

import pulp
import math
from collections import namedtuple
from copy import deepcopy
from random import randrange

import cvxpy as cp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])
def dots(customers, facilities, solution):
	x = []
	y = []


	plt.clf()
	for u in customers:
		x.append(u.location.x)
		y.append(u.location.y)
	xc = np.array(x)
	yc = np.array(y)
	plt.scatter(xc, yc, marker ='o', c = 'y', s=10, cmap='viridis')
	plt.draw()
	x = []
	y = []
	trans = []
	for u in facilities:
		x.append(u.location.x)
		y.append(u.location.y)
		if solution.count(u.index) > 0:
			trans.append('m')
		else:
			trans.append('c')
	xf = np.array(x)
	yf = np.array(y)
	plt.scatter(xf, yf, c = trans, marker = 'p', s=30, cmap='viridis')
	plt.draw()
	plt.show()
	plt.pause(0.00001)
def lindraw(customers, facilities, solution, cc):
	plt.clf()
	dots(customers, facilities, solution)
	for i in range(cc):
		if solution[i] >= 0:
			plt.plot([facilities[solution[i]].location.x, customers[i].location.x], [facilities[solution[i]].location.y, customers[i].location.y], linewidth=0.3, color='r', alpha = 0.7)
	plt.draw()
	plt.show()
	plt.pause(0.00003)

def length(point1, point2):
	return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
def object(customers, facilities, solution, used):
	obj = sum([f.setup_cost*used[f.index] for f in facilities])
	for customer in customers:
		obj += length(customer.location, facilities[solution[customer.index]].location)
	return obj
def objectMem(facilities, solution, used, disM):
	obj = sum([f.setup_cost*used[f.index] for f in facilities])
	for i in range(len(solution)):
		obj += disM[solution[i]][i]
	return obj

def solve_it(input_data):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = input_data.split('\n')

	parts = lines[0].split()
	facility_count = int(parts[0])
	customer_count = int(parts[1])
	
	facilities = []
	for i in range(1, facility_count+1):
		parts = lines[i].split()
		facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

	customers = []
	for i in range(facility_count+1, facility_count+1+customer_count):
		parts = lines[i].split()
		customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

	#print ("facilities:", facility_count,"\n customers:", customer_count)
	#return 0
	solution = [-1]*len(customers)
#plotting
	x = []
	y = []

	plt.ion()
	plt.clf()
	for u in customers:
		x.append(u.location.x)
		y.append(u.location.y)
	xc = np.array(x)
	yc = np.array(y)
	plt.scatter(xc, yc, s=10, cmap='viridis')
	plt.draw()
	x = []
	y = []
	plt.show()
	plt.pause(1)
	for u in facilities:
		x.append(u.location.x)
		y.append(u.location.y)
	xf = np.array(x)
	yf = np.array(y)
	plt.scatter(xf, yf, s=30, cmap='viridis')
	plt.draw()
	plt.show()
	dots(customers, facilities, solution)
	plt.pause(0.0001)


# build a trivial solution
# pack the facilities one by one until all the customers are served
	if facility_count > 200:
		
		capacity_remaining = [f.capacity for f in facilities]

		facility_index = 0
		for customer in customers:
			if capacity_remaining[facility_index] >= customer.demand:
				solution[customer.index] = facility_index
				capacity_remaining[facility_index] -= customer.demand
			else:
				facility_index += 1
				assert capacity_remaining[facility_index] >= customer.demand
				solution[customer.index] = facility_index
				capacity_remaining[facility_index] -= customer.demand
			print(customer, " and", facility_index)
		used = [0]*len(facilities)
		for facility_index in solution:
			used[facility_index] = 1
		
		used = [0]*len(facilities)
		for facility_index in solution:
			used[facility_index] = 1
		lindraw(customers, facilities, solution, customer_count)
	
#creating distant matrix
	disM = []
	for i in range(facility_count):

		disM.append([])
		for j in range(customer_count):
			disM[i].append(length(customers[j].location, facilities[i].location))
	#print (disM)
#pulp

	
	if facility_count < 500:

		lindraw(customers, facilities, solution, customer_count)
		
		des = pulp.LpVariable.dicts('xx', [str(i) for i in range(facility_count)], 0, 1, cat=pulp.LpInteger)
		print (des)
		ind = [str(i) for i in range(facility_count)]
		jnd = [str(j)for j in range(customer_count)]
		Routes = [(w,b) for w in ind for b in jnd]
		print (Routes)
		MMM = pulp.LpVariable.dicts('yy', (ind,jnd), 0, 1, cat=pulp.LpInteger)
		#print (MMM)
		#print (MMM['1']['2'])
		#print (des['1'])
		loc = pulp.LpProblem("facloc", pulp.LpMinimize)

		for j in range(customer_count):
			for i in range(facility_count):
				if i == solution[j]:
					MMM[str(i)][str(j)].setInitialValue(1)
					des[str(i)].setInitialValue(1)
				else:
					MMM[str(i)][str(j)].setInitialValue(0)
				
					
	#capacity constraint:
		for i in range(facility_count):
			loc += pulp.lpSum([customers[j].demand*MMM[str(i)][str(j)] for j in range(customer_count)]) <= facilities[i].capacity*des[str(i)], "Capacity %s"%i


	#loneliness:
		for j in range(customer_count):
			loc += pulp.lpSum([MMM[str(i)][str(j)] for i in range(facility_count)]) == 1, "loneliness %s"%j
	
		loc += pulp.lpSum([facilities[i].setup_cost*des[str(i)] for i in range(facility_count)]) + pulp.lpSum([MMM[str(i)][str(j)]*disM[i][j] for i in range(facility_count) for j in range(customer_count)]), "minimem_cost"
		
		if customer_count == 50:
			timlim = 500
			gap = 0.001
		elif customer_count == 200:
			timlim = 1000
			gap = 0.001
		elif customer_count == 100:
			timlim = 2000
			gap = 0.001
		elif customer_count == 1000:
			timlim = 10000
			gap = 0.00001
		elif customer_count == 800:
			timlim = 10000
			gap = 0.00001
		else:
			timlim = 6000
			gap = 0.001

		solver = pulp.PULP_CBC_CMD(msg=1, mip_start=1, maxSeconds = timlim, fracGap = gap)
		#solver = pulp.GUROBI_CMD(msg=1, mip_start=1)
		print (loc.variables)
		status = loc.solve(solver)
		print(pulp.LpStatus[status])
		if pulp.LpStatus[status] == "Optimal":
			opt = 1
		else:
			opt = 0
		solutionPulp = []
		for j in range(customer_count):
			for i in range(facility_count):
				if MMM[str(i)][str(j)].value() == 1:
					solutionPulp.append(i)
		used = []
		for i in range(facility_count):
			used.append(des[str(i)].value())
		print (solutionPulp)

		lindraw(customers, facilities, solutionPulp, customer_count)

		solution = solutionPulp
		
	#print ("PULP", objPulp, "vs", lv, "CP")
# calculate the cost of the solution
	#obj = object(customers, facilities, solution, used)
	obj = objectMem(facilities, solution, used, disM)

	#plt.ioff()
	plt.show()
	plt.pause(30)
	plt.close()
# prepare the solution in the specified output format
	output_data = '%.2f' % obj + ' ' + str(0) + '\n'
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
		print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
