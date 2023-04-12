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
import matplotlib._color_data as mcd

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])
def dots(customers):
	x = []
	y = []


	plt.clf()
	for u in customers:
		x.append(u.x)
		y.append(u.y)
	xc = np.array(x)
	yc = np.array(y)
	plt.scatter(xc, yc, marker ='o', c = 'y', s=10, cmap='viridis')
	plt.scatter([customers[0].x], [customers[0].y], marker ='o', c = 'r', s=15, cmap='viridis')
	plt.draw()
	x = []
	y = []
	plt.show()
	plt.pause(0.0001)
def lindraw(customers, vehicle_tours, V):
	plt.clf()
	#dots(customers)
	#print(vehicle_tours)
	overlap = [name for name in mcd.XKCD_COLORS]
	for i in range(V):
		
		for j in range(len(vehicle_tours[i])):
			plt.plot([vehicle_tours[i][j%len(vehicle_tours[i])].x, vehicle_tours[i][(j+1)%len(vehicle_tours[i])].x], [vehicle_tours[i][j%len(vehicle_tours[i])].y, vehicle_tours[i][(j+1)%len(vehicle_tours[i])].y], linewidth=2, color=overlap[i%len(overlap)], alpha = 1)
	x = []
	y = []
	#plt.pause(3)

	for u in customers:
		x.append(u.x)
		y.append(u.y)
	xc = np.array(x)
	yc = np.array(y)
	plt.scatter(xc, yc, marker ='o', c = 'y', s=10, cmap='viridis')
	plt.scatter([customers[0].x], [customers[0].y], marker ='o', c = 'r', s=15, cmap='viridis')
	plt.draw()
	plt.show()
	plt.pause(0.00003)
def linsimple(cust1, cust2, n):
	overlap = [name for name in mcd.XKCD_COLORS]
	plt.plot([cust1.x, cust2.x], [cust1.y, cust2.y], linewidth=2, color=overlap[n%len(overlap)], alpha = 1)
	plt.draw()
	plt.show()
	plt.pause(0.00003)
def length(customer1, customer2):
	return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)
def distMatrix(customers):
	cc = len(customers)
	resmat = np.ndarray((cc, cc))
	for i in range(cc):
		for j in range(i, cc):
			if i == j:
				resmat[i][j] = 0
			else:
				resmat[i][j] = length(customers[i], customers[j])
				resmat[j][i] = length(customers[i], customers[j])
	return resmat
def objMem(customers, vehicle_tours, vehicle_count, distance):
	obj = 0
	for v in range(vehicle_count):
		for i in range(len(vehicle_tours[v])):
			obj += distance[vehicle_tours[v][i].index][vehicle_tours[v][(i+1)%len(vehicle_tours[v])].index]
	return obj

def randpath(ndcnt):
	domain = list(range(0, ndcnt))
	sol = []

	for i in range(ndcnt):
		nex = randrange(len(domain))
		sol.append(domain[nex])
		domain.pop(nex)
	return sol
def pathlen(sol, nodeCount, points):
	jobj = length(points[sol[-1]], points[sol[0]])
	for index in range(0, nodeCount-1):
		jobj += length(points[sol[index]], points[sol[index+1]])
	return jobj
def greedy(nodeCount, points):
	temp_sol = []
	des = [0]*nodeCount
	temp_sol.append(0)
	des[0] = 1
	for i in range(nodeCount - 1):
		temp_sol.append(0)
		l = 999999999999999999999999999
		print ("/////////////////////////////////////////////////////////////////////////////////////")
		for j in range(nodeCount):
			if des[j] == 0:
				ll = length(points[temp_sol[i]], points[j])
				print (i,"/", nodeCount, "..",j,"..)", ll, "and", l)
				if ll < l:
					l = ll
					temp_sol[i+1] = j
		des[temp_sol[i+1]] = 1

	
	return temp_sol

def rswap(solution, nodeCount):
	sol = deepcopy(solution)
	sol.append(sol[0])
	init = randrange(nodeCount)
	lenn = randrange(nodeCount - init + 1)
	for i in range(init, init + (lenn+1)//2):
		sol[i], sol[init + lenn - i] = sol[init + lenn - i], sol[i]
	sol.pop()
	return sol
def intersec(point1, point2, point3, point4):
#is there an intersec
	if point4 == point1 or point1 == point2 or point2 == point3 or point2 == point4:
		return False
	if length(point1, point2) == 0:
		return False
	else:
		asr = math.asin(math.copysign(1, point2.x-point1.x)*(point2.y - point1.y)/(length(point1, point2)))
	if length(point3, point1) == 0:
		return False
	else:
		as1 = math.asin(math.copysign(1, point3.x-point1.x)*(point3.y - point1.y)/(length(point1, point3)))
	if length(point1, point4) == 0:
		return False
	else:
		as2 = math.asin(math.copysign(1, point4.x-point1.x)*(point4.y - point1.y)/(length(point1, point4)))
	if (point2.x - point1.x) < 0:
		asr -= math.pi
	if (point3.x - point1.x) < 0:
		as1 -= math.pi
	if (point4.x - point1.x) < 0:
		as2 -= math.pi
	flag = 0
	if asr < as2 and asr > as1 or asr < as1 and asr > as2:
		flag = 1
		#print("ass", asr, as1, as2)
	elif asr == as2:
		if length(point2, point1) > length(point1, point4):
			return True
	elif asr == as1:
		if length(point2, point1) > length(point1, point3):
			return True
	else:
		return False
	if length(point3, point4) == 0:
		return False
	else:
		asr = math.asin((point4.y - point3.y)/(length(point3, point4)))
	if length(point3, point1) == 0:
		return False
	else:
		as1 = math.asin((point1.y - point3.y)/(length(point1, point3)))
	if length(point3, point2) == 0:
		return False
	else:
		as2 = math.asin((point2.y - point3.y)/(length(point2, point3)))
	if (point4.x - point3.x) < 0:
		asr -= math.pi
	if (point1.x - point3.x) < 0:
		as1 -= math.pi
	if (point2.x - point3.x) < 0:
		as2 -= math.pi
	if (asr < as2 and asr > as1 or asr < as1 and asr > as2) and flag == 1:
		#print("ass2", asr, as1, as2)
		return True
	elif asr == as2:
		if length(point2, point1) > length(point1, point4):
			return True
	elif asr == as1:
		if length(point2, point1) > length(point1, point3):
			return True
	else:
		return False
def interremoveS(solution, nodeCount, points):
#removes intersections
	sol = deepcopy(solution)
	t = deepcopy(sol)
	
	for i in range(nodeCount - 1):
		sol.append(sol[0])
		for j in range(i+2, nodeCount ):
			if intersec(points[sol[i]], points[sol[i+1]], points[sol[j]], points[sol[j+1]]):
				#print (sol[i], sol[j])
				swap(sol, i, j)
				#print (sol)
		sol.pop()
		if pathlen(sol, nodeCount, points) < pathlen(t, nodeCount, points):
			t = deepcopy(sol)
		else:
			sol = deepcopy(t)
	return t
def swap(sol, i, j):
	for k in range(i+1, i+1 + (j-i)//2):
		sol[k], sol[j - k + i + 1] = sol[j - k + i + 1], sol[k]
def iterrat(temp_sol, solution, nodeCount, tobj, cunt, points):
	#print(tobj, ":)", "::", cunt)
	
	ct = 40
	temp_sol = rswap(temp_sol, nodeCount)
	#print (pathlen(temp_sol, nodeCount, points))

	if pathlen(temp_sol, nodeCount, points) < tobj:
		solution = []
		solution.extend(temp_sol)
		print ("fuckshit")

		return solution
	elif cunt < ct:
		
		temp_sol = iterrat(temp_sol, solution, nodeCount, tobj, cunt + 1, points)
		return temp_sol
	else:
		return solution

def shift(lst, steps):
	if steps < 0:
		steps = abs(steps)
		for i in range(steps):
			lst.append(lst.pop(0))
	else:
		for i in range(steps):
			lst.insert(0, lst.pop())

def k_pop(solution, nodeCount, p1, points):
#some kind of k-opt
	temp_sol = deepcopy(solution)
	cunt = 0
	sol = deepcopy(solution)
	tabu = []
	opt = pathlen(solution, nodeCount, points)
	print("opt", opt)
	while cunt < math.sqrt(nodeCount) and len(tabu) < nodeCount - 1:

		if p1 == nodeCount-1:
			mm = length(points[temp_sol[p1]], points[temp_sol[0]])
			p2 = 0
		else:
			mm = length(points[temp_sol[p1]], points[temp_sol[p1+1]])
			p2 = p1 + 1
		p4 = p1
		#print ("p1 is", p1)
		for i in range(nodeCount):
			if i != p1 and i != p2 and i != p2+1 and tabu.count(temp_sol[i]) == 0 and mm > length(points[temp_sol[p2]], points[temp_sol[i]]):
				p4 = i
				mm = length(points[temp_sol[p2]], points[temp_sol[i]])
		#print ("p4 is", p4)
		
		if p1 == p4:
			print("oops")
			break
		tabu.append(temp_sol[p4])
		shift(temp_sol, -1*p1)
		if p4 < p1:
			p4 = nodeCount + p4 - p1
		else:
			p4 -= p1
		p3 = p4 - 1
		p1 = 0
		p2 = 1
		
		swap(temp_sol, p1, p3)
		if pathlen(temp_sol, nodeCount, points) < opt:
			sol = []
			sol.extend(temp_sol)
			opt = pathlen(temp_sol, nodeCount, points)
			print("yes", cunt)
			print("opt", opt)
			cunt += 6
		else:
			print("no", cunt)
			cunt += 1


		
	return sol
def maxList(lst):
	res = -999999999999999
	for i in lst:
		if i > res:
			res = i
	return lst.index(res)
def minList(lst):
	res = 999999999999999
	for i in lst:
		if i < res:
			res = i
	return lst.index(res)
def pulping(customers, vehicle_count, vehicle_capacity):
	customer_count = len(customers)
	distance = distMatrix(customers)
	if 3 == 3:

		knd = [str(k) for k in range(1, vehicle_count+1)]
		ind = [str(i) for i in range(customer_count)]
		jnd = [str(j)for j in range(customer_count)]
		desX = pulp.LpVariable.dicts('x', (knd, ind, jnd), 0, 1, cat=pulp.LpInteger)
		#print(desX)
		ind.pop(0)
		desY = pulp.LpVariable.dicts('y', (knd, ind), 0, 1, cat=pulp.LpInteger)
		routes = pulp.LpProblem("routes", pulp.LpMinimize)
		demands = [customers[i].demand for i in range(customer_count)]
		#print(desY)
		#print(demands)
		connection = pulp.LpVariable.dicts('c', (knd, jnd), 0, customer_count, cat = pulp.LpInteger)
		#fuckshit = pulp.LpVariable.dicts('F', (knd, jnd, ind), 0, 1, cat = pulp.LpInteger)
		for k in range(1, 1 + vehicle_count):
			routes += connection[str(k)]['0'] == 1
			for i in range(1, customer_count):
				routes += connection[str(k)][str(i)] >= 2
				routes += connection[str(k)][str(i)] <= customer_count
				for j in range(1, customer_count):
					routes += connection[str(k)][str(i)] - connection[str(k)][str(j)] + 1 <= (customer_count -1)*(1 - desX[str(k)][str(i)][str(j)])
			routes += pulp.lpSum([desX[str(k)]['0'][str(j)] for j in range(1,customer_count)]) <= 1, "depot %s"%k
			
			routes += pulp.lpSum([demands[i]*desY[str(k)][str(i)] for i in range(1, customer_count)]) <= vehicle_capacity, "cap %s"%k
			

			for p in range(customer_count):
				routes += (pulp.lpSum([desX[str(k)][str(i)][str(p)] for i in range(customer_count)]) - pulp.lpSum([desX[str(k)][str(p)][str(j)] for j in range(customer_count)])) == 0
				routes += desX[str(k)][str(p)][str(p)] == 0


				for j in range(customer_count):
					routes += desX[str(k)][str(p)][str(j)] + desX[str(k)][str(j)][str(p)] <= 1 
				
			for i in range(1, customer_count):
			#	for j in range(customer_count):
			#		routes += fuckshit[str(k)][str(j)][str(i)] <= desX[str(k)][str(j)][str(i)]
			#		routes += fuckshit[str(k)][str(j)][str(i)] <= connection[str(j)]
			#		routes += fuckshit[str(k)][str(j)][str(i)] >= connection[str(j)] + desX[str(k)][str(j)][str(i)] - 1

				routes += pulp.lpSum([desX[str(k)][str(i)][str(j)] for j in range(customer_count)]) == desY[str(k)][str(i)]

		routes += pulp.lpSum([pulp.lpSum([desX[str(k)][str(0)][str(j)] for j in range(customer_count)]) for k in range(1, vehicle_count + 1)]) >= 1
		#loneliness
		for i in range(1, customer_count):
			routes += pulp.lpSum([desY[str(k)][str(i)] for k in range(1, vehicle_count + 1)]) == 1, "alon %s"%i
		

		for j in range(1, customer_count):
			routes += pulp.lpSum([pulp.lpSum([desX[str(k)][str(i)][str(j)] for i in range(customer_count)]) for k in range(1, 1+vehicle_count)]) == 1, "once %s"%j
		routes += pulp.lpSum([pulp.lpSum([desX[str(k)][str(i)][str(0)] for i in range(customer_count)]) for k in range(1, 1+vehicle_count)]) >= 1, "more in 0"
		routes += pulp.lpSum([pulp.lpSum([pulp.lpSum([distance[i][j]*desX[str(k)][str(i)][str(j)] for k in range(1, vehicle_count+1)]) for j in range(customer_count)]) for i in range(customer_count)])
		
		timlim = customer_count*50
		gap = 0.005
		
	
		solver = pulp.PULP_CBC_CMD(msg=1, mip_start=1, maxSeconds = timlim, fracGap = gap)
		status = routes.solve(solver)
		dots(customers)
		for k in range(1, vehicle_count+1):
			for i in range(customer_count):
				if i != 0:
					print("y", k, i, "---", desY[str(k)][str(i)].value())
				for j in range(customer_count):
					print("x", k, i, j, "___", desX[str(k)][str(i)][str(j)].value())
					if desX[str(k)][str(i)][str(j)].value() == 1:
						linsimple(customers[i], customers[j], k)
	
		#for i in range(1, customer_count):
		#	print("connection", i,"///", connection[str(i)].value())
	
		plt.show()
		plt.pause(5)
		vehicle_tours = []
		flg = []
		for k in range(1, 1+vehicle_count):
			flg.append(0)
			for i in range(1, customer_count):
				if desY[str(k)][str(i)].value() == 1:
					flg[k-1] = 1
					break
		for k in range(vehicle_count):
			vehicle_tours.append([customers[0]])
			previous = 0
			if flg[k] == 1:
				t = 0
				for i in range(customer_count):
					ggwp = 0
					for j in range(customer_count):
						if desX[str(k+1)][str(previous)][str(j)].value() == 1:
							if j != 0:
								vehicle_tours[k].append(customers[j])
								previous = j
								t += 1
								break
							else:
								ggwp = 1
								break
					if ggwp == 1:
						break
		
		
		return vehicle_tours	
def solprint(veh):
	c = 0
	for v in veh:
		print("\nvex", c,":")
		c += 1
		for i in v:
			print(i.index, end = "; ")
	print("\n")
def solve_it(input_data):
# Modify this code to run your optimization algorithm

# parse the input
	lines = input_data.split('\n')

	parts = lines[0].split()
	customer_count = int(parts[0])
	vehicle_count = int(parts[1])
	vehicle_capacity = int(parts[2])
	
	customers = []
	for i in range(1, customer_count+1):
		line = lines[i]
		parts = line.split()
		customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

#the depot is always the first customer in the input
	depot = customers[0] 

	distance = distMatrix(customers)
	#print (distance)
# build a trivial solution
	plt.ion()
	
# assign customers to vehicles starting by the largest customer demands
	
	vehicle_tours = []
	
	remaining_customers = set(customers)
	remaining_customers.remove(depot)
	for v in range(0, vehicle_count):
		# print "Start Vehicle: ",v
		vehicle_tours.append([])
		capacity_remaining = vehicle_capacity
		while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
			used = set()
			order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
			for customer in order:
				if capacity_remaining >= customer.demand:
					capacity_remaining -= customer.demand
					vehicle_tours[v].append(customer)
					# print '   add', ci, capacity_remaining
					used.add(customer)
			remaining_customers -= used
	print(customer_count, "customers, ", vehicle_count, "vehicles. and capacity is", vehicle_capacity)
	
# pulp modeling
	
	if 3 == 3:

		knd = [str(k) for k in range(1, vehicle_count+1)]
		ind = [str(i) for i in range(customer_count)]
		jnd = [str(j)for j in range(customer_count)]
		desX = pulp.LpVariable.dicts('x', (knd, ind, jnd), 0, 1, cat=pulp.LpInteger)
		#print(desX)
		ind.pop(0)
		desY = pulp.LpVariable.dicts('y', (knd, ind), 0, 1, cat=pulp.LpInteger)
		routes = pulp.LpProblem("routes", pulp.LpMinimize)
		demands = [customers[i].demand for i in range(customer_count)]
		#print(desY)
		#print(demands)
		connection = pulp.LpVariable.dicts('c', (knd, jnd), 0, customer_count, cat = pulp.LpInteger)
		for k in range(1, 1 + vehicle_count):
			routes += connection[str(k)]['0'] == 1
			for i in range(1, customer_count):
				routes += connection[str(k)][str(i)] >= 2
				routes += connection[str(k)][str(i)] <= customer_count
				for j in range(1, customer_count):
					routes += connection[str(k)][str(i)] - connection[str(k)][str(j)] + 1 <= (customer_count -1)*(1 - desX[str(k)][str(i)][str(j)])
			routes += pulp.lpSum([desX[str(k)]['0'][str(j)] for j in range(1,customer_count)]) <= 1, "depot %s"%k
			
			routes += pulp.lpSum([demands[i]*desY[str(k)][str(i)] for i in range(1, customer_count)]) <= vehicle_capacity, "cap %s"%k
			

			for p in range(customer_count):
				routes += (pulp.lpSum([desX[str(k)][str(i)][str(p)] for i in range(customer_count)]) - pulp.lpSum([desX[str(k)][str(p)][str(j)] for j in range(customer_count)])) == 0
				routes += desX[str(k)][str(p)][str(p)] == 0


				for j in range(customer_count):
					routes += desX[str(k)][str(p)][str(j)] + desX[str(k)][str(j)][str(p)] <= 1 
				
			for i in range(1, customer_count):
			
				routes += pulp.lpSum([desX[str(k)][str(i)][str(j)] for j in range(customer_count)]) == desY[str(k)][str(i)]

		routes += pulp.lpSum([pulp.lpSum([desX[str(k)][str(0)][str(j)] for j in range(customer_count)]) for k in range(1, vehicle_count + 1)]) >= 1
		#loneliness
		for i in range(1, customer_count):
			routes += pulp.lpSum([desY[str(k)][str(i)] for k in range(1, vehicle_count + 1)]) == 1, "alon %s"%i
		

		for j in range(1, customer_count):
			routes += pulp.lpSum([pulp.lpSum([desX[str(k)][str(i)][str(j)] for i in range(customer_count)]) for k in range(1, 1+vehicle_count)]) == 1, "once %s"%j
		routes += pulp.lpSum([pulp.lpSum([desX[str(k)][str(i)][str(0)] for i in range(customer_count)]) for k in range(1, 1+vehicle_count)]) >= 1, "more in 0"
		routes += pulp.lpSum([pulp.lpSum([pulp.lpSum([distance[i][j]*desX[str(k)][str(i)][str(j)] for k in range(1, vehicle_count+1)]) for j in range(customer_count)]) for i in range(customer_count)])
		if vehicle_count == 3:
			timlim = 500
			gap = 0.001
		elif vehicle_count == 8:
			timlim = 6000
			gap = 0.0001
		elif vehicle_count == 5:
			timlim = 8000
			gap = 0.0001
		elif vehicle_count == 10:
			timlim = 8000
			gap = 0.0001
		elif vehicle_count == 16:
			timlim = 7500
			gap = 0.0001
		elif vehicle_count == 41:
			timlim = 9000
			gap = 0.0001
		else 
			timlim = 6000
			gap = 0.001
	if vehicle_count > 1000:
		solver = pulp.PULP_CBC_CMD(msg=1, mip_start=1, maxSeconds = timlim, fracGap = gap)
		status = routes.solve(solver)
		dots(customers)
		for k in range(1, vehicle_count+1):
			for i in range(customer_count):
				if i != 0:
					print("y", k, i, "---", desY[str(k)][str(i)].value())
				for j in range(customer_count):
					print("x", k, i, j, "___", desX[str(k)][str(i)][str(j)].value())
					if desX[str(k)][str(i)][str(j)].value() == 1:
						linsimple(customers[i], customers[j], k)
	
		#for i in range(1, customer_count):
		#	print("connection", i,"///", connection[str(i)].value())
	
		plt.show()
		plt.pause(5)
		vehicle_tours = []
		flg = []
		for k in range(1, 1+vehicle_count):
			flg.append(0)
			for i in range(1, customer_count):
				if desY[str(k)][str(i)].value() == 1:
					flg[k-1] = 1
					break
		for k in range(vehicle_count):
			vehicle_tours.append([customers[0]])
			if flg[k] == 1:
				t = 0
				for i in range(customer_count):
					ggwp = 0
					for j in range(customer_count):
						if desX[str(k+1)][str(vehicle_tours[k][t].index)][str(j)].value() == 1:
							if j != 0:
								vehicle_tours[k].append(customers[j])
								t += 1
								break
							else:
								ggwp = 1
								break
					if ggwp == 1:
						break
		
		#print(vehicle_tours)
		if vehicle_count == 3:
			f = open('1.txt', 'w')
		elif vehicle_count == 8:
			f = open('2.txt', 'w')
		elif vehicle_count == 5:
			f = open('3.txt', 'w')
		elif vehicle_count == 10:
			f = open('4.txt', 'w')
		elif vehicle_count == 16:
			f = open('5.txt', 'w')
		elif vehicle_count == 41:
			f = open('6.txt', 'w')
		for v in vehicle_tours:
			for i in v:
				f.write(str(i.index)+' ')
			f.write('\n')
		f.close()
		objf = 99999999999999
	
	

#reading from file
#local search starts
	vehicle_tours = []
	if vehicle_count == 3:
		f = open('1.txt', 'r')
	elif vehicle_count == 8:
		f = open('2.txt', 'r')
	elif vehicle_count == 5:
		f = open('3.txt', 'r')
	elif vehicle_count == 10:
		f = open('4.txt', 'r')
	elif vehicle_count == 16:
		f = open('5.txt', 'r')
	elif vehicle_count == 41:
		f = open('6.txt', 'r')
	l = [line.strip() for line in f]
	f.close()
#put in vehicle_tours from file
	t = 0
	for v in l:
		#solver = pulp.PULP_CBC_CMD(msg=1, mip_start=1, maxSeconds = timlim, fracGap = gap)
		separated = v.split()
		vehicle_tours.append([])
		gg = 0
		for i in separated:
			vehicle_tours[t].append(customers[int(i)])
			gg += 1
		t += 1
	for v in vehicle_tours:
		shift(v, -1*v.index(customers[0]))
	objf = objMem(customers, vehicle_tours, vehicle_count, distance)
	print("obj in file ", objf)	

	impflag = 0
	for ii in range(0):
		vccounter = 0
		objes = []
		for vc in vehicle_tours:
			print("v", vccounter)
			nodeCount = len(vc)
			
			points = []
			filesolution = []
		
#tsp begins here
			t = 0
			for i in vc:
				points.append(i)
				filesolution.append(t)
				#solcust.append(i)
				t += 1
	#greedy
			grsol = greedy(nodeCount, points)
			if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
				finsol = deepcopy(grsol)
			else:
				finsol = deepcopy(filesolution)
				grsol = deepcopy(filesolution)
			obj = pathlen(finsol, nodeCount, points)
			bobj = obj
	#random improvement
			for l in range(20//(1+round(math.sqrt(nodeCount)))):
				solution = deepcopy(grsol)
				tobj = obj
				for i in range(300//(1+round(math.sqrt(nodeCount)))):
					print (l, "...", i, "vehicle", vccounter, ")", bobj)
					solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
					tobj = pathlen(solution, nodeCount, points)	


				if tobj < bobj:
					finsol = deepcopy(solution)
					bobj = tobj
			solution = deepcopy(finsol)
			obj = bobj
	#intersections remove
			for i in range(600//(1+round(math.sqrt(nodeCount)))):
				print("vehicle", vccounter)
				print (i,"Sstage")
				
				solution = interremoveS(solution, nodeCount, points)
				if obj == pathlen(solution, nodeCount, points):
					break
				obj = pathlen(solution, nodeCount, points)
				print(obj)
	#k-pop
			tabu = []
			for j in range(round(50*math.pow(nodeCount, 0.25))):
				print("vehicle", vccounter)	

				mm = length(points[solution[-1]], points[solution[0]])
				p1 = nodeCount - 1
				
				for i in range(nodeCount - 1):
					if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
						p1 = i
						tabu.append(solution[p1])
						mm = length(points[solution[i]], points[solution[i+1]])		


				print (j, end = ":")
				print (solution)
				solution = k_pop(solution, nodeCount, p1, points)

			obj = pathlen(solution, nodeCount, points)
	#intersections remove again
			for i in range(600//(1+round(math.sqrt(nodeCount)))):
				print("vehicle", vccounter)
				print (i,"Sstage")
				
				solution = interremoveS(solution, nodeCount, points)
				if obj == pathlen(solution, nodeCount, points):
					break
				obj = pathlen(solution, nodeCount, points)
				print(obj)
			objes.append(obj)
			for i in range(len(vc)):
				vc[i] = points[solution[i]]
			lindraw(customers, [vc], 1)
			plt.show()
			plt.pause(0.5)
			vccounter += 1
#pulp helps
		for v in vehicle_tours:
			shift(v, -1*v.index(customers[0]))
		if 3!=3:
			print(objes)
			if impflag == 1:
				objes.pop(maxindex)
			maxindex = maxList(objes)
			minindex = minList(objes)
			rememberme = objes[minindex]+objes[maxindex]
			print("before", rememberme)
			print("max", maxindex,"; min", minindex)
			solprint([vehicle_tours[maxindex],vehicle_tours[minindex]])
			
			cust = []
			for i in vehicle_tours[maxindex]:
				cust.append(i)
			for i in range(1, len(vehicle_tours[minindex])):
				cust.append(vehicle_tours[minindex][i])

			temper_tours = pulping(cust, 2, vehicle_capacity)
			print ("obj is", objMem(customers, temper_tours, 2, distance))
			solprint(temper_tours)
			plt.pause(6)
			if round(objMem(customers, temper_tours, 2, distance)) < round(rememberme):
				print("improved, but....")
				#do it again
				vccounter = 0
				for vc in temper_tours:
					print("v", vccounter)
					nodeCount = len(vc)
			
					points = []
					filesolution = []
					t = 0
					for i in vc:
						points.append(i)
						filesolution.append(t)
						#solcust.append(i)
						t += 1
			#greedy
					grsol = greedy(nodeCount, points)
					if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
						finsol = deepcopy(grsol)
					else:
						finsol = deepcopy(filesolution)
						grsol = deepcopy(filesolution)
					obj = pathlen(finsol, nodeCount, points)
					bobj = obj
			#random improvement
					for l in range(60//(1+round(math.sqrt(nodeCount)))):
						solution = deepcopy(grsol)
						tobj = obj
						for i in range(200//(1+round(math.sqrt(nodeCount)))):
							print (l, "...", i, "vehicle", vccounter, ")", bobj)
							solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
							tobj = pathlen(solution, nodeCount, points)	


						if tobj < bobj:
							finsol = deepcopy(solution)
							bobj = tobj
					solution = deepcopy(finsol)
					obj = bobj
			#intersections remove
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(100*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
					
					for i in range(len(vc)):
						vc[i] = points[solution[i]]
					lindraw(customers, [vc], 1)
					plt.show()
					plt.pause(2)
					vccounter += 1
				for v in temper_tours:
					shift(v, -1*v.index(customers[0]))
				print("improved to ", objMem(customers, temper_tours, 2, distance))
				lindraw(customers, temper_tours, 2)
				plt.show()
				plt.pause(5)
				vehicle_tours[maxindex] = deepcopy(temper_tours[0])
				vehicle_tours[minindex] = deepcopy(temper_tours[1])
				for v in vehicle_tours:
					shift(v, -1*v.index(customers[0]))
				impflag = 0
			else:
				print ("didn't improve.....yet")
		#do it again
				vccounter = 0
				for vc in temper_tours:
					print("v", vccounter)
					nodeCount = len(vc)
			
					points = []
					filesolution = []
					t = 0
					for i in vc:
						points.append(i)
						filesolution.append(t)
						#solcust.append(i)
						t += 1
			#greedy
					grsol = greedy(nodeCount, points)
					if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
						finsol = deepcopy(grsol)
					else:
						finsol = deepcopy(filesolution)
						grsol = deepcopy(filesolution)
					obj = pathlen(finsol, nodeCount, points)
					bobj = obj
			#random improvement
					for l in range(10//(1+round(math.sqrt(nodeCount)))):
						solution = deepcopy(grsol)
						tobj = obj
						for i in range(300//(1+round(math.sqrt(nodeCount)))):
							print (l, "...", i, "vehicle", vccounter, ")", bobj)
							solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
							tobj = pathlen(solution, nodeCount, points)	


						if tobj < bobj:
							finsol = deepcopy(solution)
							bobj = tobj
					solution = deepcopy(finsol)
					obj = bobj
			#intersections remove
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(70*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(90*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
					for i in range(len(vc)):
						vc[i] = points[solution[i]]
					lindraw(customers, [vc], 1)
					vccounter += 1
				for v in temper_tours:
					shift(v, -1*v.index(customers[0]))
				if round(objMem(customers, temper_tours, 2, distance)) < round(rememberme):
					print("vsyo-taki improved", objMem(customers, temper_tours, 2, distance))
					lindraw(customers, temper_tours, 2)

					plt.show()
					plt.pause(5)
					vehicle_tours[maxindex] = deepcopy(temper_tours[0])
					vehicle_tours[minindex] = deepcopy(temper_tours[1])
					for v in vehicle_tours:
						shift(v, -1*v.index(customers[0]))
					impflag = 1
				else:
					print ("didn't improve(((")	
					impflag = 1
		if objMem(customers, vehicle_tours, vehicle_count, distance) <= 630.009 and vehicle_count == 8:
			break
				
#random))))
		if 1 == 1:
			maxindex = maxList(objes)
			rmaxindex = maxindex
			rminindex = rmaxindex
			while rminindex == rmaxindex:
				rminindex = randrange(vehicle_count)
			rememberme = objes[rminindex]+objes[rmaxindex]
			print("before", rememberme)
			print("mrax", rmaxindex,"; mrin", rminindex)
			solprint([vehicle_tours[rmaxindex],vehicle_tours[rminindex]])
			
			cust = []
			for i in vehicle_tours[rmaxindex]:
				cust.append(i)
			for i in range(1, len(vehicle_tours[rminindex])):
				cust.append(vehicle_tours[rminindex][i])

			temper_tours = pulping(cust, 2, vehicle_capacity)
			print ("obj is", objMem(customers, temper_tours, 2, distance))
			solprint(temper_tours)
			plt.pause(6)
			if round(objMem(customers, temper_tours, 2, distance)) < round(rememberme):
				print("random improved lol")
				#do it again
				vccounter = 0
				for vc in temper_tours:
					print("v", vccounter)
					nodeCount = len(vc)
			
					points = []
					filesolution = []
					t = 0
					for i in vc:
						points.append(i)
						filesolution.append(t)
						#solcust.append(i)
						t += 1
			#greedy
					grsol = greedy(nodeCount, points)
					if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
						finsol = deepcopy(grsol)
					else:
						finsol = deepcopy(filesolution)
						grsol = deepcopy(filesolution)
					obj = pathlen(finsol, nodeCount, points)
					bobj = obj
			#random improvement
					for l in range(30//(1+round(math.sqrt(nodeCount)))):
						solution = deepcopy(grsol)
						tobj = obj
						for i in range(600//(1+round(math.sqrt(nodeCount)))):
							print (l, "...", i, "vehicle", vccounter, ")", bobj)
							solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
							tobj = pathlen(solution, nodeCount, points)	


						if tobj < bobj:
							finsol = deepcopy(solution)
							bobj = tobj
					solution = deepcopy(finsol)
					obj = bobj
			#intersections remove
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
					for i in range(len(vc)):
						vc[i] = points[solution[i]]
					lindraw(customers, [vc], 1)
					vccounter += 1
				for v in temper_tours:
					shift(v, -1*v.index(customers[0]))
				print(" random improved to ", objMem(customers, temper_tours, 2, distance))
				lindraw(customers, temper_tours, 2)

				plt.show()
				plt.pause(5)
				vehicle_tours[rmaxindex] = deepcopy(temper_tours[0])
				vehicle_tours[rminindex] = deepcopy(temper_tours[1])
				for v in vehicle_tours:
					shift(v, -1*v.index(customers[0]))
				impflag = 0
			else:
				print ("didn't improve by random))....yeeeet")
		#do it again
				vccounter = 0
				for vc in temper_tours:
					print("v", vccounter)
					nodeCount = len(vc)
			
					points = []
					filesolution = []
					t = 0
					for i in vc:
						points.append(i)
						filesolution.append(t)
						#solcust.append(i)
						t += 1
			#greedy
					grsol = greedy(nodeCount, points)
					if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
						finsol = deepcopy(grsol)
					else:
						finsol = deepcopy(filesolution)
						grsol = deepcopy(filesolution)
					obj = pathlen(finsol, nodeCount, points)
					bobj = obj
			#random improvement
					for l in range(30//(1+round(math.sqrt(nodeCount)))):
						solution = deepcopy(grsol)
						tobj = obj
						for i in range(600//(1+round(math.sqrt(nodeCount)))):
							print (l, "...", i, "vehicle", vccounter, ")", bobj)
							solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
							tobj = pathlen(solution, nodeCount, points)	


						if tobj < bobj:
							finsol = deepcopy(solution)
							bobj = tobj
					solution = deepcopy(finsol)
					obj = bobj
			#intersections remove
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
					
					for i in range(len(vc)):
						vc[i] = points[solution[i]]
					lindraw(customers, [vc], 1)
					plt.show()
					plt.pause(5)
					vccounter += 1
				for v in temper_tours:
					shift(v, -1*v.index(customers[0]))
				if round(objMem(customers, temper_tours, 2, distance)) < round(rememberme):
					print("vsyo-taki random improved")
					lindraw(customers, temper_tours, 2)

					vehicle_tours[rmaxindex] = deepcopy(temper_tours[0])
					vehicle_tours[rminindex] = deepcopy(temper_tours[1])
					for v in vehicle_tours:
						shift(v, -1*v.index(customers[0]))
					impflag = 0
				else:
					print ("random didn't improve(((")	
					impflag = 1
		if 1 == 1:
			if impflag == 1:
				rmidindex = maxindex
			else:
				rmaxindex = randrange(vehicle_count)

			rminindex = rmaxindex
			while rminindex == rmaxindex:
				rminindex = randrange(vehicle_count)
			rmidindex = rminindex
			while rmidindex == rminindex or rmidindex == rmaxindex:
				rmidindex = randrange(vehicle_count)
			rememberme = objes[rminindex]+objes[rmaxindex] + objes[rmidindex]
			print("before", rememberme)
			print("mrax", rmaxindex,"; mrin", rminindex, "; mrid", rmidindex)
			solprint([vehicle_tours[rmaxindex],vehicle_tours[rminindex], vehicle_tours[rmidindex]])
			
			cust = []
			for i in vehicle_tours[rmaxindex]:
				cust.append(i)
			for i in range(1, len(vehicle_tours[rminindex])):
				cust.append(vehicle_tours[rminindex][i])
			for i in range(1, len(vehicle_tours[rmidindex])):
				cust.append(vehicle_tours[rmidindex][i])

			temper_tours = pulping(cust, 3, vehicle_capacity)
			print ("obj is", objMem(customers, temper_tours, 3, distance))
			solprint(temper_tours)
			plt.pause(6)
			if round(objMem(customers, temper_tours, 3, distance)) < round(rememberme):
				print("random improved lol")
				#do it again
				vccounter = 0
				for vc in temper_tours:
					print("v", vccounter)
					nodeCount = len(vc)
			
					points = []
					filesolution = []
					t = 0
					for i in vc:
						points.append(i)
						filesolution.append(t)
						#solcust.append(i)
						t += 1
			#greedy
					grsol = greedy(nodeCount, points)
					if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
						finsol = deepcopy(grsol)
					else:
						finsol = deepcopy(filesolution)
						grsol = deepcopy(filesolution)
					obj = pathlen(finsol, nodeCount, points)
					bobj = obj
			#random improvement
					for l in range(6//(1+round(math.sqrt(nodeCount)))):
						solution = deepcopy(grsol)
						tobj = obj
						for i in range(60//(1+round(math.sqrt(nodeCount)))):
							print (l, "...", i, "vehicle", vccounter, ")", bobj)
							solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
							tobj = pathlen(solution, nodeCount, points)	


						if tobj < bobj:
							finsol = deepcopy(solution)
							bobj = tobj
					solution = deepcopy(finsol)
					obj = bobj
			#intersections remove
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
					for i in range(len(vc)):
						vc[i] = points[solution[i]]
					lindraw(customers, [vc], 1)
					vccounter += 1
				for v in temper_tours:
					shift(v, -1*v.index(customers[0]))
				print(" 3-random improved to ", objMem(customers, temper_tours, 3, distance))
				lindraw(customers, temper_tours, 3)

				plt.show()
				plt.pause(5)
				vehicle_tours[rmaxindex] = deepcopy(temper_tours[0])
				vehicle_tours[rmidindex] = deepcopy(temper_tours[1])
				vehicle_tours[rminindex] = deepcopy(temper_tours[2])
				for v in vehicle_tours:
					shift(v, -1*v.index(customers[0]))
				impflag = 0
			else:
				print ("didn't improve by 3-random))....yeeeet")
		#do it again
				vccounter = 0
				for vc in temper_tours:
					print("v", vccounter)
					nodeCount = len(vc)
			
					points = []
					filesolution = []
					t = 0
					for i in vc:
						points.append(i)
						filesolution.append(t)
						#solcust.append(i)
						t += 1
			#greedy
					grsol = greedy(nodeCount, points)
					if pathlen(grsol, nodeCount, points) < pathlen(filesolution, nodeCount, points):
						finsol = deepcopy(grsol)
					else:
						finsol = deepcopy(filesolution)
						grsol = deepcopy(filesolution)
					obj = pathlen(finsol, nodeCount, points)
					bobj = obj
			#random improvement
					for l in range(8//(1+round(math.sqrt(nodeCount)))):
						solution = deepcopy(grsol)
						tobj = obj
						for i in range(100//(1+round(math.sqrt(nodeCount)))):
							print (l, "...", i, "vehicle", vccounter, ")", bobj)
							solution = iterrat(solution, solution, nodeCount, tobj, 0, points)
							tobj = pathlen(solution, nodeCount, points)	


						if tobj < bobj:
							finsol = deepcopy(solution)
							bobj = tobj
					solution = deepcopy(finsol)
					obj = bobj
			#intersections remove
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
			#k-pop
					tabu = []
					for j in range(round(50*math.pow(nodeCount, 0.25))):
						print("vehicle", vccounter)	

						mm = length(points[solution[-1]], points[solution[0]])
						p1 = nodeCount - 1
						
						for i in range(nodeCount - 1):
							if tabu.count(solution[i]) == 0 and mm < length(points[solution[i]], points[solution[i+1]]):
								p1 = i
								tabu.append(solution[p1])
								mm = length(points[solution[i]], points[solution[i+1]])		


						print (j, end = ":")
						print (solution)
						solution = k_pop(solution, nodeCount, p1, points)

					obj = pathlen(solution, nodeCount, points)
			#intersections remove again
					for i in range(600//(1+round(math.sqrt(nodeCount)))):
						print("vehicle", vccounter)
						print (i,"Sstage")
					
						solution = interremoveS(solution, nodeCount, points)
						if obj == pathlen(solution, nodeCount, points):
							break
						obj = pathlen(solution, nodeCount, points)
						print(obj)
					for i in range(len(vc)):
						vc[i] = points[solution[i]]
					lindraw(customers, [vc], 1)
					plt.show()
					plt.pause(5)
					vccounter += 1
				for v in temper_tours:
					shift(v, -1*v.index(customers[0]))
				if round(objMem(customers, temper_tours, 3, distance)) < round(rememberme):
					print("vsyo-taki 3-random improved")
					lindraw(customers, temper_tours, 3)

					vehicle_tours[rmaxindex] = deepcopy(temper_tours[0])
					vehicle_tours[rmidindex] = deepcopy(temper_tours[1])
					vehicle_tours[rminindex] = deepcopy(temper_tours[2])
					for v in vehicle_tours:
						shift(v, -1*v.index(customers[0]))
					impflag = 0
				else:
					print ("3-random didn't improve(((")	
					impflag = 1
					for v in vehicle_tours:
						shift(v, -1*v.index(customers[0]))
				
#calculate obj
		
			#for i in range(1, customer_count):
			#	print("connection", i,"///", connection[str(i)].value())
	for v in vehicle_tours:
		shift(v, -1*v.index(customers[0]))
	lindraw(customers, vehicle_tours, vehicle_count)

	plt.show()
	plt.pause(2)
#useless 
	'''
			vehicle_tours = []
			flg = []
			for k in range(1, 1+vehicle_count):
				flg.append(0)
				for i in range(1, customer_count):
					if desY[str(k)][str(i)].value() == 1:
						flg[k-1] = 1
						break
			for k in range(vehicle_count):
				vehicle_tours.append([customers[0]])
				if flg[k] == 1:
					t = 0
					for i in range(customer_count):
						ggwp = 0
						for j in range(customer_count):
							if desX[str(k+1)][str(vehicle_tours[k][t].index)][str(j)].value() == 1:
								if j != 0:
									vehicle_tours[k].append(customers[j])
									t += 1
									break
								else:
									ggwp = 1
									break
						if ggwp == 1:
							break
	'''
		
		#print(vehicle_tours)
		#print(l)
# checks that the number of customers served is correct
	assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1 + vehicle_count
# calculate the cost of the solution; for each vehicle the length of the route

	obj = objMem(customers, vehicle_tours, vehicle_count, distance)
#xz
	
	print("obj ", obj)
	if objf >= obj:
		#print(vehicle_tours)
		if vehicle_count == 3:
			f = open('1.txt', 'w')
		elif vehicle_count == 8:
			f = open('2.txt', 'w')
		elif vehicle_count == 5:
			f = open('3.txt', 'w')
		elif vehicle_count == 10:
			f = open('4.txt', 'w')
		elif vehicle_count == 16:
			f = open('5.txt', 'w')
		elif vehicle_count == 41:
			f = open('6.txt', 'w')
		for v in vehicle_tours:
			for i in v:
				f.write(str(i.index)+' ')
			f.write('\n')
		f.close()

		print("e boy")
		#objf = obj


	'''
	for v in range(0, vehicle_count):
		vehicle_tour = vehicle_tours[v]
		if len(vehicle_tour) > 0:
			obj += length(depot,vehicle_tour[0])
			for i in range(0, len(vehicle_tour)-1):
				obj += length(vehicle_tour[i],vehicle_tour[i+1])
			obj += length(vehicle_tour[-1],depot)
	'''
# prepare the solution in the specified output format
	lindraw(customers, vehicle_tours, vehicle_count)
	outputData = '%.2f' % obj + ' ' + str(0) + '\n'
	for v in range(0, vehicle_count):
		outputData += ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
	#plt.ioff()
	plt.show()
	plt.pause(30)
	plt.close()
	return outputData


import sys

if __name__ == '__main__':
	import sys
	if len(sys.argv) > 1:
		file_location = sys.argv[1].strip()
		with open(file_location, 'r') as input_data_file:
			input_data = input_data_file.read()
		print(solve_it(input_data))
	else:

		print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

