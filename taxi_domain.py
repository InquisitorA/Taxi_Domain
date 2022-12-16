import csv
import json
import random
import sys
import time

import numpy as np


sys.setrecursionlimit(10000)

# SE,SN,SW,SS are used to maintain the information about the grid whether it can move in respective directions

SE = [[0 for i in range(5)] for j in range(5)]
SW = [[0 for i in range(5)] for j in range(5)]
SN = [[0 for i in range(5)] for j in range(5)]
SS = [[0 for i in range(5)] for j in range(5)]

for i in range(0,5):
	for j in range(0,5):
		if i==0:
			SN[i][j] = -1
		if i==4:
			SS[i][j] = -1
		if j==0:
			SW[i][j] = -1
		if j==4:
			SE[i][j] = -1

		if j==1:
			if i==0 or i==1:
				SE[i][j] = -1
		if j==2:
			if i==0 or i==1:
				SW[i][j] = -1
		if j==0:
			if i==3 or i==4:
				SE[i][j] = -1

		if j==2:
			if i==3 or i==4:
				SE[i][j] = -1

		if j==1:
			if i==3 or i==4:
				SW[i][j] = -1

		if j==3:
			if i==3 or i==4:
				SW[i][j] = -1




class State:
	def __init__(self,px,py,tx,ty,status,value):
		self.px = px
		self.py = py
		self.tx = tx
		self.ty = ty
		self.status = status
		self.value = value

	def setval(self,value):
		self.value = value

V = []
for px in range(5):
	for py in range(5):
		for tx in range(5):
			for ty in range(5):
				G = State(px,py,tx,ty,0,0)
				V.append(G)

for px in range(5):
	for py in range(5):
		G = State(px,py,px,py,1,0)
		V.append(G)


source = [4,0]
taxi  = [0,0]
dest  = [0,4]

qns = input("Enter the qn no: ")


def nextstate(V,k,action,gamma):
	
	G = V[k]
	px = G.px
	
	py = G.py
	tx = G.tx
	ty = G.ty
	status = G.status
	i = tx
	j = ty
	if action=="East":
		if SE[i][j] == 0:
			return gamma*V[k+1].value-1
		else:
			return gamma*V[k].value-1

	if action == "West":
		if SW[i][j] == 0:
			return gamma*V[k-1].value-1
		else:
			return gamma*V[k].value-1

	if action == "North":
		if SN[i][j]==0:
			return gamma*V[k-5].value-1
		else:
			return gamma*V[k].value-1	

	if action == "South":
		if SS[i][j]==0:
			return gamma*V[k+5].value-1
		else:
			return gamma*V[k].value-1	

	if action == "Pickup":
		if status==1:
			return gamma*V[k].value-1
		else:
			k1 = 625 + 5*tx + ty
			if px==tx and py==ty:
				return gamma*V[k1].value-1
			else:
				return gamma*V[k].value-10

	if action == "Putdown":
		if status==0:
			return gamma*V[k].value-10
		else:
			k1 = 125*px + 25*py + 5*tx + ty

			if px==dest[0] and py==dest[1]:
				return 20
			else:
				return gamma*V[k1].value-1



#these functions returns the value if a probable action is taken


D = ["East","West","North","South","Pickup","Putdown"]

def Returnval(V,k,action,gamma):
	
	J = []
	for i in range(6):
		J.append(nextstate(V,k,D[i],gamma))

	if action=="East":
		return (0.85*J[0])+(0.05*J[1])+(0.05*J[2])+(0.05*J[3])
	elif action=="West":
		return (0.05*J[0])+(0.85*J[1])+(0.05*J[2])+(0.05*J[3])
	elif action=="North":
		return (0.05*J[0])+(0.05*J[1])+(0.85*J[2])+(0.05*J[3])
	elif action=="South":
		return (0.05*J[0])+(0.05*J[1])+(0.05*J[2])+(0.85*J[3])
	elif action=="Pickup":
		return J[4]
	else:
		return J[5]





def givenext(V,k,action):
	G = V[k]
	px = G.px
	
	py = G.py
	tx = G.tx
	ty = G.ty
	status = G.status
	i = tx
	j = ty
	if action=="East":
		if SE[i][j] == 0:
			return k+1
		else:
			return k
	if action == "West":
		if SW[i][j] == 0:
			return k-1
		else:
			return k
	if action == "North":
		if SN[i][j]==0:
			return k-5
		else:
			return k	
	if action == "South":
		if SS[i][j]==0:
			return k+5
		else:
			return k	
	if action == "Pickup":
		if status==1:
			return k
		else:
			k1 = 625 + 5*tx + ty
			if px==tx and py==ty:
				return k1
			else:
				return k
	if action == "Putdown":
		if status==0:
			return k
		else:
			k1 = 125*px + 25*py + 5*tx + ty
			if px==dest[0] and py==dest[1]:
				return -1000
			else:
				return k1


def printpath(V,Act):
	obtained = False
	k = 125*source[0] + 25*source[1]+ 5*taxi[0]+taxi[1]

	lu = 0
	while(lu<200):
		if k==-1000:
			print("found " + str(lu))
			break
		else:
			print(Act[k] + " " + str(k))
			k = givenext(V,k,Act[k])
		
		lu = lu + 1


def equallist(L1,L2):
	for i in range(650):
		if L1[i]!=L2[i]:
			return False
	return True




def valueitera(V,gamma,epsil):
	count = 0
	maxd = 1000
	VD = [-1 for i in range(650)]
	A = []
	L1 = []
	while(maxd>epsil):
		maxdiff = -10000
	
		for k in range(650):
			
			V1 = V.copy()
			a = V[k].value
			L = []
			for i in range(6):
				L.append(Returnval(V1,k,D[i],gamma))

			maxS = max(L)
			V[k].setval(maxS) 
			maxind = L.index(maxS)
			VD[k] = D[maxind]
			difference = abs(maxS - a)

			if maxdiff<difference:
				maxdiff = difference
		count = count + 1
		maxd = maxdiff
		L1.append(maxd)
		
		
	A.append(count)
	A.append(VD)
	A.append(L1)
	

	return A




Actit = []

if qns == "1a":
	gamma = float(input("Enter the discount factor: "))
	epsil = float(input("Enter the max error allowed: "))
	Actions = valueitera(V,gamma,epsil)
	Actit = Actions[1]
	maxdi = Actions[2]
	q = input("Enter subpart: ")
	if q=="a":
		print(str(epsil) + " " + str(Actions[0]))
	if q=="b":
		print(str(gamma) + " " + str(Actions[0]))
	if q == "c":
		print("discount: " + str(gamma))
		for i in range(len(maxdi)):
			print(str(maxdi[i])   )

	if q == "d":
		printpath(V,Actit)




for i in range(650):
	V[i].setval(0)






def policyiter1(V,gamma):

	Act = ["North" for i in range(650)]
	Rec = [-1 for i in range(650)]
	L1 = []
	count = 0

	while (not equallist(Act,Rec)):
		V1 = V.copy()
		V2 = []
		for i in range(650):
			V2.append(V[i].value)

		maxd = 100000
		s = 0
		while(maxd>0.001):
			maxd = -1000
			for k in range(650):
				a = V[k].value
				V[k].setval(Returnval(V1,k,Act[k],gamma))
				if maxd<abs(V[k].value-a):
					maxd = abs(V[k].value-a)
			s = s+1
			
		PA = []
		for k in range(650):
			PA.append(V[k].value)
		L1.append(PA)
		

	
		Rec = Act.copy()

		for k in range(650):
			L = []
			for i in range(6):
				L.append(Returnval(V,k,D[i],gamma))

			maxS = max(L)

			maxind = L.index(maxS)
			Act[k] = D[maxind]
		
		count = count + 1
	print(count)

	Actions = []
	Actions.append(L1)
	Actions.append(count)
	Actions.append(Act)

	return Actions


Actpol1 = []
if qns == "1b":
	gamma = float(input("Enter the discount factor: "))
	start = time.time()
	Actions = policyiter1(V,gamma)
	end = time.time()
	print(str(end-start) + " secs")

	h = len(Actions[0])
	
	for i in range(len(Actions[0])-1):
		maxd = -1000
		for j in range(650):
			p = abs(Actions[0][i][j] - Actions[0][h-1][j])
			if p > maxd:
				maxd = p
		print(maxd)


for i in range(650):
	V[i].setval(0)






def possiblestates(V,k,action):
	L = []
	if action == "Pickup":
		k1 = givenext(V,k,action)
		L.append(k1)

		return L

	elif action == "Putdown":
		k1 = givenext(V,k,action)
		if k1==-1000:
			return L
		L.append(k1)
		return L

	else:
		L.append(givenext(V,k,"East"))
		L.append(givenext(V,k,"West"))
		L.append(givenext(V,k,"North"))
		L.append(givenext(V,k,"South"))
		return L







def policyiter2(V,gamma):

	Act = ["North" for i in range(650)]
	Rec = [-1 for i in range(650)]

	count = 0
	while (not equallist(Act,Rec)):
		
		A = np.zeros((650,650),dtype="float64")
		B = np.zeros(650,dtype = "float64")

		for i in range(650):
			A[i][i] = 1

		for i in range(650):
			h = D.index(Act[i])
			r = Returnval(V,i,Act[i],0)
			B[i] = r
			L = possiblestates(V,i,Act[i])
			if len(L)==1:
				k1 = L[0]
				A[i][k1] = A[i][k1] - gamma
			elif len(L)==4:
				for u in range(4):
					if u==h:
						A[i][L[u]] = A[i][L[u]] - (0.85*gamma)
					else:
						A[i][L[u]] = A[i][L[u]] - (0.05*gamma)


		X = np.dot(np.linalg.inv(A),B.reshape(650,))
		Vi = X.reshape(650).tolist()

		for i in range(650):
			V[i].setval(Vi[i])

		Rec = Act.copy()

		for k in range(650):
			L = []
			for i in range(6):
				L.append(Returnval(V,k,D[i],gamma))

			maxS = max(L)

			maxind = L.index(maxS)
			Act[k] = D[maxind]
		
		count = count + 1
		print(count)
	
	print(count)
	return Act



Actpol2 = []
if qns == "1c":
	gamma = float(input("Enter the discount factor: "))
	start = time.time()
	Actpol2 = policyiter2(V,gamma)
	end = time.time()
	print(str(end-start) + " secs")




#print(equallist(Actit,Actpol1))




##to print the path from initial position to destination




#PART----B------------------------------------------------------------------------------------------------------------
















D = ["East","West","North","South","Pickup","Putdown"]

def simulator(action):

	AS = ["East","West","North","South"]

	
	
	if action=="Pickup"  or action == "Putdown":
		return action

	else:
		h = AS.index(action)
		r = random.randrange(0,99)
		
		if r<85:
			return action
			
		if r>=85 and r<90:
			return AS[(h+1)%4]
						
		if r>=90 and r<95:
			return AS[(h+2)%4]
			
		if r>=95 and r<100:
			return AS[(h+3)%4]
			


def rewardsimulator(V,k,action):


	G = V[k]
	px = G.px
	
	py = G.py
	tx = G.tx
	ty = G.ty
	status = G.status

	if action == "Pickup":

		if status==0 and px==tx and py==ty:
			return -1
		elif status==0:
			return -10
		else:
			return -1

	elif action == "Putdown":
		if status ==1 and tx==dest[0] and ty == dest[1]:
			return 20
		elif status == 1:
			return -1
		else:
			return -10

	else:
		return -1






def qlearning(V,gamma,alpha,epsilon,decay):
	Q = []

	for i in range(650):
		L = [0,0,0,0,0,0]
		Q.append(L)

	e = epsilon
	step = 0

	r = 0

	while(step<5000):
		kt = random.randrange(0,649)
		p = 0
		e1 = e
		if decay==1:
			e1 = e/(step+1)
		rewardx = 0
		while(p<500):

			i = Q[kt].index(max(Q[kt]))
			

			gy = random.randrange(1,100000)
			if gy<(100000*e1):
				i =  random.randrange(0,5)

			action = simulator(D[i])

			ac = D.index(action)

			kt1 = givenext(V,kt,action)
			reward = rewardsimulator(V,kt,action)
			
			rewardx = rewardx + (pow(gamma,p)*reward)
			
			if kt1==-1000:
				Q[kt][ac] = (1-alpha)*Q[kt][ac] + (alpha*(reward))
				break
				
			Q[kt][ac] = ((1-alpha)*Q[kt][ac]) + (alpha*(reward+gamma*max(Q[kt1])))
			kt = kt1
			p = p+1
		step = step + 1
		r = r + rewardx
		if step%10==0:
			print(str(r/10))
			r = 0
		
	return Q





if qns == "2a":

	gamma = float(input("Enter the discount: "))
	alpha = float(input("Enter the learning rate: "))
	epsilon = float(input("Enter the exploration rate: "))
	decay = float(input("Enter decay status: "))

	Q = qlearning(V,gamma,alpha,epsilon,decay)
	policy = []
	
	for i in range(650):
		policy.append(D[Q[i].index(max(Q[i]))])
	#printpath(V,policy)





def sarsalearning(V,gamma,alpha,epsilon,decay):
	Q = []

	for i in range(650):
		L = [0,0,0,0,0,0]
		Q.append(L)

	e = epsilon
	step = 0
	r = 0

	while(step<5000):
		kt = random.randrange(0,649)
		p = 0
		e1 = e
		if decay==1:
			e1 = e/(step+1)


		rewardx = 0
		while(p<500):

			i = Q[kt].index(max(Q[kt]))
			

			gy = random.randrange(1,100000)
			if gy<(100000*e1):
				i =  random.randrange(0,5)

			action = simulator(D[i])

			ac = D.index(action)

			kt1 = givenext(V,kt,action)
			reward = rewardsimulator(V,kt,action)
			rewardx = rewardx + (pow(gamma,p)*reward)

			if kt1==-1000:
				Q[kt][ac] = (1-alpha)*Q[kt][ac] + (alpha*(reward))
				break

			i1 = Q[kt1].index(max(Q[kt1]))
			gy = random.randrange(1,100000)
			if gy<(100000*e1):
				i1 =  random.randrange(0,5)
			action1 = simulator(D[i1])
			p2 = D.index(action1)


				
			Q[kt][ac] = ((1-alpha)*Q[kt][ac]) + (alpha*(reward+gamma*(Q[kt1][p2])))
			kt = kt1
			p = p+1
		
		
		step = step + 1
		r = r + rewardx
		if step%10==0:
			print(str(r/10))
			r = 0
	
	return Q






if qns == "2b":

	gamma = float(input("Enter the discount: "))
	alpha = float(input("Enter the learning rate: "))
	epsilon = float(input("Enter the exploration rate: "))
	decay = float(input("Enter the decay status: "))


	Q = sarsalearning(V,gamma,alpha,epsilon,decay)

	policy = []
	
	for i in range(650):
		policy.append(D[Q[i].index(max(Q[i]))])







"""
	obtained = False

	k = 125*source[0] + 25*source[1]+ 5*taxi[0]+taxi[1]
	
	print(policy[k])
	
	step = 0
	reward = 0
	while(step<200):

		if k==-1000:
			print("found " + str(step))
			break
		else:
			print(policy[k])
			reward = reward + (pow(0.9,step)*rewardsimulator(V,k,policy[k]))
			k1 = k
			k = givenext(V,k,policy[k])

			if k1==k:
				print("iywjoslp0oh")
			
		
		step = step + 1
	print(reward)


"""








