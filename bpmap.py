# 2D Pose Graph Belief Propagation
# Andrew Davison, Imperial College London, 2019

import numpy as np
import math
import time
import pygame
import random
from pygame.locals import *
pygame.init()


# set the width and height of the screen (pixels)
WIDTH = 1000
HEIGHT = 1000

size = [WIDTH, HEIGHT]
black = (0,0,0)
green = (0,255,0)
lightblue = (0,180,255)
darkblue = (0,40,160)
red = (255,100,0)
white = (255,255,255)
blue = (0,0,255)
grey = (110,110,110)




# Screen centre will correspond to (x, y) = (0, 0)
u0 = 100
v0 = HEIGHT - 100

POINTSXRANGE = 10.0
POINTSYRANGE = 10.0
NUMBEROFPOINTS = 20
NUMBEROFFACTORS = 50

kx = (WIDTH - 200) / float(POINTSXRANGE)

# For printing text
myfont = pygame.font.SysFont("Jokerman", 40)

# Standard deviations
MEASUREMENTsigma = 0.5
POSEsigma = 2.0

# For laying out measurements in simulation: only connect points up to this distance apart
MAXMEASDISTANCE = 4.0

# For batch solution
bigSigma = -1
bigmu = -1


pygame.init()



# Initialise Pygame display screen
screen = pygame.display.set_mode(size)
# This makes the normal mouse pointer invisible in graphics window
#pygame.mouse.set_visible(0)




class GroundTruthPoint:
    def __init__(self, index):
        self.index = index
        x = random.uniform (0.0, POINTSXRANGE)
        y = random.uniform (0.0, POINTSXRANGE)
        self.x = np.matrix([[x],[y]])
        print ("New ground truth point", index, "with position", x, y)



class GroundTruthMeasurement:
    def __init__(self, fromgroundtruthpoint, togroundtruthpoint):
        self.fromgroundtruthpoint = fromgroundtruthpoint
        self.togroundtruthpoint = togroundtruthpoint
        self.ztrue = togroundtruthpoint.x - fromgroundtruthpoint.x
        self.znoisy = self.ztrue.copy()
        self.znoisy[0,0] += random.uniform(-MEASUREMENTsigma, MEASUREMENTsigma)
        self.znoisy[1,0] += random.uniform(-MEASUREMENTsigma, MEASUREMENTsigma)
        print ("New factor from point", fromgroundtruthpoint.index, "to point", togroundtruthpoint.index)
        print ("Ground truth", self.ztrue, "Noisy", self.znoisy)






class VariableNode:
    def __init__(self, variableID):
        self.variableID = variableID
        # mu and Sigma are stored for visualisation purposes, etc., but are not the master data here
        self.mu = np.matrix([[0.0], [0.0]])
        self.Sigma = np.matrix([[10.0, 0.0], [0.0, 10.0]])

        # List of links to edges
        self.edges = []
        self.updatedflag = 0

    def draw(self):
        if (self.updatedflag == 1):
            drawcolour = green
        else:
            drawcolour = lightblue
        pygame.draw.circle(screen, drawcolour, (int(u0 + kx * self.mu[0,0]), int(v0 - kx * self.mu[1,0])), 4, 0)
        sigma = math.sqrt(self.Sigma[0,0])
        pygame.draw.circle(screen, lightblue, (int(u0 + kx * self.mu[0,0]), int(v0 - kx * self.mu[1,0])), int(sigma * kx), 1)

    def sendAllMessages(self):
        #print("Sending all messages: could be implemented much more efficiently.")
        for (i, edge) in enumerate(self.edges):
            self.sendMessage(i)

    def updatemuSigma(self):
        # Just update the mu mean estimate from all current incoming messsages
        eta = np.matrix([[0.0], [0.0]])
        Lambdaprime = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # Multiply inward messages from other factors
        for i, edge in enumerate(self.edges):
            # Multiply incoming messages by adding precisions
            (etainward, Lambdaprimeinward) = (self.edges[i].factortovariablemessage.eta, self.edges[i].factortovariablemessage.Lambda)
            eta += etainward
            Lambdaprime += Lambdaprimeinward

        self.Sigma = Lambdaprime.I
        self.mu = self.Sigma * eta



    # localfactorID is factor we send message out to
    def sendMessage(self, outedgeID):
        #print ("Sending message from variable", self.variableID, "to edge with local ID", outedgeID, "global ID", self.edges[outedgeID].edgeID)
        eta = np.matrix([[0.0], [0.0]])
        Lambdaprime = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # Multiply inward messages from other factors
        for i, edge in enumerate(self.edges):
            if (i != outedgeID):
                # Multiply incoming messages by adding precisions
                #print ("Multiplying inward message from edge with local ID", i, "global ID", edge.edgeID)
                (etainward, Lambdaprimeinward) = (self.edges[i].factortovariablemessage.eta, self.edges[i].factortovariablemessage.Lambda)
                eta += etainward
                Lambdaprime += Lambdaprimeinward
                #print ("eta, Lambdaprime", eta, Lambdaprime)
        self.edges[outedgeID].variabletofactormessage.eta = eta.copy()
        self.edges[outedgeID].variabletofactormessage.Lambda = Lambdaprime.copy()
        #print (                self.edges[outedgeID].VariableToFactorMessage)
        #print ("Message", self.edges[outedgeID].VariableToFactorMessage[0],self.edges[outedgeID].VariableToFactorMessage[1])

        # Update local state estimate from all messages including outward one
        eta += self.edges[outedgeID].factortovariablemessage.eta
        Lambdaprime += self.edges[outedgeID].factortovariablemessage.Lambda
        if (Lambdaprime[0,0] != 0.0):
            self.Sigma = Lambdaprime.I
        self.mu = self.Sigma * eta
        self.edges[outedgeID].variabletofactormessage.mu = self.mu.copy()


class VariableToFactorMessage:
    def __init__(self, dimension):
        self.eta = np.zeros((dimension, 1))
        self.Lambda = np.zeros((dimension, dimension))
        self.mu = np.zeros((dimension, 1))

class FactorToVariableMessage:
    def __init__(self, dimension):
        self.eta = np.zeros((dimension, 1))
        self.Lambda = np.zeros((dimension, dimension))


# Every edge between a variable node and a factor node is via an edge
# So the variable and factor only need to know about the edge, not about
# each other's internals
# An edge stores the recent message in each direction; dimension is the state space size
class Edge:
    def __init__(self, dimension):
        # VariableToFactorMessage also passes the current state value (3rd arg)
        self.variabletofactormessage = VariableToFactorMessage(dimension)
        self.factortovariablemessage = FactorToVariableMessage(dimension)
        self.variable = -1
        self.factor = -1

# What we need to specify for a factor:
# Number of variables inputs and size of each: implicit in edges
# Measurement (vector with size)
# h (function of x)
# J (function of x)
# Lambda matrix
# Write generalised code for marginalisation
class FactorNode:
    def __init__(self, factorID, edges, z):
        self.factorID = factorID
        self.edges = edges
        self.z = z
        self.etastored = -1
        self.Lambdaprimestored = -1
        self.recalculateetaLambdaprime()
        print ("New FactorNode factorID = ", factorID)

    # Get current overall stacked state vector estimate from input edges
    def x0(self):
        for (i, edge) in enumerate(self.edges):
            if (i==0):
                x0local = edge.variabletofactormessage.mu
            else:
                x0local = np.concatenate((x0local, edge.variabletofactormessage.mu), axis=0)
        return x0local


    def sendAllMessages(self):
        #print("Sending all messages: could be implemented much more efficiently.")
        for (i,edge) in enumerate(self.edges):
            self.sendMessage(i)

    # Relinearise factor if needed (with linear factor only need to call once)
    def recalculateetaLambdaprime(self):
        # First form eta and Lambdaprime for linearised factor
        # Recalculate h and J in principle so we can relinearise
        hlocal = self.h()
        Jlocal = self.J()
        self.Lambdaprimestored = Jlocal.T * self.Lambda * Jlocal
        self.etastored = Jlocal.T * self.Lambda * (Jlocal * self.x0() + self.z - hlocal)


    def sendMessage(self, outedgeID):
        #print ("Sending message from factor", self.factorID, "to edge with local ID", outedgeID)

        # Copy factor precision vector and matrix from stored; could recalculate every time
        # here if needed.
        eta = self.etastored.copy()
        Lambdaprime = self.Lambdaprimestored.copy()

        # Next condition factors on messages from all edges apart from outward one
        offset = 0
        outoffset = 0
        outsize = 0
        for (i, edge) in enumerate(self.edges):
            edgeeta = edge.variabletofactormessage.eta
            edgeLambda = edge.variabletofactormessage.Lambda
            edgesize = edgeeta.shape[0]
            if (i != outedgeID):
                # For edges which are not the outward one, condition
                eta[offset:offset+edgesize,:] += edgeeta
                Lambdaprime[offset:offset+edgesize,offset:offset+edgesize] += edgeLambda
            else:
                # Remember where in the matrix the outward one is
                outoffset = offset
                outsize = edgesize
            offset += edgesize

        # Now restructure eta and Lambdaprime so the outward variable of interest is at the top
        etatoptemp = eta[0:outsize,:].copy()
        eta[0:outsize,:] = eta[outoffset:outoffset+outsize,:]
        eta[outoffset:outoffset+outsize,:] = etatoptemp
        Lambdaprimetoptemp = Lambdaprime[0:outsize,:].copy()
        Lambdaprime[0:outsize,:] = Lambdaprime[outoffset:outoffset+outsize,:]
        Lambdaprime[outoffset:outoffset+outsize,:] = Lambdaprimetoptemp
        Lambdaprimelefttemp = Lambdaprime[:,0:outsize].copy()
        Lambdaprime[:,0:outsize] = Lambdaprime[:,outoffset:outoffset+outsize]
        Lambdaprime[:,outoffset:outoffset+outsize] = Lambdaprimelefttemp


        # To marginalise, first set up subblocks as in Eustice
        ea = eta[0:outsize,:]
        eb = eta[outsize:,:]
        aa = Lambdaprime[0:outsize,0:outsize]
        ab = Lambdaprime[0:outsize,outsize:]
        ba = Lambdaprime[outsize:,0:outsize]
        bb = Lambdaprime[outsize:,outsize:]

        bbinv = bb.I
        emarg = ea - ab * bbinv * eb
        Lmarg = aa - ab * bbinv * ba

        for row in range(outsize):
            self.edges[outedgeID].factortovariablemessage.eta[row,0] = emarg[row,0]
            for col in range(outsize):
                self.edges[outedgeID].factortovariablemessage.Lambda[row,col] = Lmarg[row,col]


# Factor for relative measurement between points
class TwoDMeasFactorNode(FactorNode):
    def __init__(self, factorID, edges, groundtruthmeasurement):
        # Calls constructor of parent class
        self.Lambda = np.matrix(  [[1.0 / (MEASUREMENTsigma * MEASUREMENTsigma), 0.0], [0.0, 1.0 / (MEASUREMENTsigma * MEASUREMENTsigma)]]  )
        super().__init__(factorID, edges, groundtruthmeasurement.znoisy)

    def h(self):
        #print ("Function to generate h.")
        xfrom = self.edges[0].variabletofactormessage.mu
        xto = self.edges[1].variabletofactormessage.mu
        h = xto - xfrom
        return h

    def J(self):
        #print ("Function to generate J.")
        # In general J could depend on variables but here it is constant because the measurement function is linear
        J = np.matrix( [[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]] )
        return J

    def setLambda(self, newMEASUREMENTsigma):
        self.Lambda = np.matrix(  [[1.0 / (newMEASUREMENTsigma * newMEASUREMENTsigma), 0.0], [0.0, 1.0 / (newMEASUREMENTsigma * newMEASUREMENTsigma)]]  )

# Simple factor for pose anchor measurement of a single point
class TwoDPoseFactorNode(FactorNode):
    def __init__(self, factorID, edges, poseobserved, Psigma):
        # Calls constructor of parent class
        self.Lambda = np.matrix(  [[1.0 / (Psigma * Psigma), 0.0], [0.0, 1.0 / (Psigma * Psigma)]]  )
        super().__init__(factorID, edges, poseobserved)

    def h(self):
        #print ("Function to generate h.")
        x = self.edges[0].variabletofactormessage.mu
        h = x
        return h

    def J(self):
        #print ("Function to generate J.")
        # In general J could depend on variables but here it is constant because the measurement function is linear
        J = np.matrix( [[1.0, 0.0], [0.0, 1.0]] )
        return J



# Just for instructions display
fflag = 1
cflag = 1
hlflag = 0



def updateDisplay():

    #Graphics
    screen.fill(black)


    for groundtruthfactor in groundtruthfactors:
        xfrom = groundtruthfactor.fromgroundtruthpoint.x[0,0]
        yfrom = groundtruthfactor.fromgroundtruthpoint.x[1,0]
        xto = groundtruthfactor.togroundtruthpoint.x[0,0]
        yto = groundtruthfactor.togroundtruthpoint.x[1,0]
        tonoisy = groundtruthfactor.fromgroundtruthpoint.x + groundtruthfactor.znoisy
        xtonoisy = tonoisy[0,0]
        ytonoisy = tonoisy[1,0]
        pygame.draw.line(screen, grey, ( int(u0 + kx * xfrom), int(v0 - kx * yfrom) ), ( int(u0 + kx * xto), int(v0 - kx * yto)), 2)
        #Draw ground noisy measurement
        #pygame.draw.line(screen, red, ( int(u0 + kx * xfrom), int(v0 - kx * yfrom) ), ( int(u0 + kx * xtonoisy), int(v0 - kx * ytonoisy)), 2)


    for groundtruthpoint in groundtruthpoints:
        pygame.draw.circle(screen, white, ( int(u0 + kx * groundtruthpoint.x[0,0]), int( v0 - kx * groundtruthpoint.x[1,0] ) ), 6, 0)


    # Draw optimal batch solution
    for (i,variablenode) in enumerate(variablenodes):
        (bx, by) = (bigmu[2*i,0],bigmu[2*i+1,0])
        sigma = math.sqrt(bigSigma[2*i,2*i])
        pygame.draw.circle(screen, green, (int(u0 + kx * bx), int(v0 - kx * by)), 4, 0)
        pygame.draw.circle(screen, green, (int(u0 + kx * bx), int(v0 - kx * by)), int(sigma * kx), 1)




    for variablenode in variablenodes:
        variablenode.draw()

    if (cflag):
        ssshow = myfont.render("Click: variable sends messages", True, white)
        screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 20))
    if (fflag):
        ssshow = myfont.render("f: start random schedule", True, white)
        screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 60))
    if (hlflag):
        ssshow = myfont.render("h,l change: measurement sigma %02f" %MEASUREMENTsigma, True, white)
        screen.blit(ssshow, (WIDTH - 20 - ssshow.get_width(), 100))

    pygame.display.flip()
    #time.sleep(0.2)





groundtruthpoints = []
groundtruthfactors = []
variablenodes = []
factornodes = []
posefactornodes = []


# Main loop


for i in range(NUMBEROFPOINTS):
    groundtruthpoints.append(GroundTruthPoint(i))
    variablenodes.append(VariableNode(i))

    poseedges = []
    newposeedge = Edge(2)
    poseedges.append(newposeedge)
    if(i == 0):
        psigma = POSEsigma / 20.0
    else:
        psigma = POSEsigma
    newposefactornode = TwoDPoseFactorNode(i, poseedges, groundtruthpoints[i].x, psigma)
    posefactornodes.append(newposefactornode)
    variablenodes[i].edges.append(newposeedge)
    newposeedge.variable = variablenodes[i]
    newposeedge.factor = newposefactornode

for i in range(NUMBEROFFACTORS):
    fromindex = random.randint(0, NUMBEROFPOINTS - 1)
    while (True):
        toindex = random.randint(0, NUMBEROFPOINTS - 1)
        newgroundtruthmeasurement = GroundTruthMeasurement(groundtruthpoints[fromindex], groundtruthpoints[toindex])
        print ("ztrue")
        print(newgroundtruthmeasurement.ztrue)
        print ("norm")
        print (np.linalg.norm(newgroundtruthmeasurement.ztrue))
        if (np.linalg.norm(newgroundtruthmeasurement.ztrue) < MAXMEASDISTANCE and fromindex != toindex):
            break
    print (fromindex, toindex)
    groundtruthfactors.append(newgroundtruthmeasurement)
    # Set up factor node
    newedges = []
    fromedge = Edge(2)
    toedge = Edge(2)
    newedges.append(fromedge)
    newedges.append(toedge)
    newtwodmeasfactornode = TwoDMeasFactorNode(i, newedges, newgroundtruthmeasurement)
    factornodes.append(newtwodmeasfactornode)

    variablenodes[fromindex].edges.append(fromedge)
    variablenodes[toindex].edges.append(toedge)
    fromedge.variable = variablenodes[fromindex]
    fromedge.factor = newtwodmeasfactornode
    toedge.variable = variablenodes[toindex]
    toedge.factor = newtwodmeasfactornode

# Testing

for posefactornode in posefactornodes:
    posefactornode.sendMessage(0)

for variablenode in variablenodes:
    variablenode.sendMessage(0)






print ("Setup Done")






# Implement optimal batch solution
bigLambda = np.zeros((2 * NUMBEROFPOINTS, 2 * NUMBEROFPOINTS))
bigeta = np.zeros((2 * NUMBEROFPOINTS, 1))

for (i,posefactornode) in enumerate(posefactornodes):
    bigeta[2*i:2*i+2,:] = posefactornode.etastored.copy()
    bigLambda[2*i:2*i+2,2*i:2*i+2] = posefactornode.Lambdaprimestored.copy()

    print (bigeta)
    print (bigLambda)


for (i,factornode) in enumerate(factornodes):
    fromindex = factornode.edges[0].variable.variableID
    toindex = factornode.edges[1].variable.variableID
    print ("fromindex", fromindex, "toindex", toindex)
    etablock = factornode.etastored.copy()
    Lambdablock = factornode.Lambdaprimestored.copy()
    print (etablock)
    print (Lambdablock)
    bigeta[2*fromindex:2*fromindex+2,:] += etablock[0:2,:]
    bigeta[2*toindex:2*toindex+2,:] += etablock[2:4,:]
    bigLambda[2*fromindex:2*fromindex+2,2*fromindex:2*fromindex+2] += Lambdablock[0:2,0:2]
    bigLambda[2*fromindex:2*fromindex+2,2*toindex:2*toindex+2] += Lambdablock[0:2,2:4]
    bigLambda[2*toindex:2*toindex+2,2*fromindex:2*fromindex+2] += Lambdablock[2:4,0:2]
    bigLambda[2*toindex:2*toindex+2,2*toindex:2*toindex+2] += Lambdablock[2:4,2:4]


    print (bigeta)
    print (bigLambda)



    bigSigma = np.matrix(bigLambda).I.copy()
    bigmu = bigSigma * bigeta

    print (bigmu)
    print (bigSigma)







updateDisplay()
updateDisplay()

count = 0


flag = 1
while(flag):
    Eventlist = pygame.event.get()

    # event handling
    for event in Eventlist:
        if (event.type == MOUSEBUTTONDOWN):
            (mousex, mousey) = pygame.mouse.get_pos()
            print ("Click at", mousex, mousey)
            x = float(mousex - u0)/kx
            y = float(-mousey + v0)/kx
            print ("Coordinates", x, y)


            thresh = 0.2
            for variablenode in variablenodes:
                dist = math.sqrt( (variablenode.mu[0,0] - x)**2 + (variablenode.mu[1,0] - y)**2 )
                if (dist < thresh):
                    variablenode.updatedflag = 1
                    variablenode.sendAllMessages()
                    # This is all very inefficient with a lot of repeated work!
                    for edge in variablenode.edges:
                        edge.factor.sendAllMessages()
                        for edge2 in edge.factor.edges:
                            edge2.variable.updatemuSigma()
                else:
                    variablenode.updatedflag = 0


        if (event.type == KEYDOWN):
            if (event.key == K_f):
                flag = 0
            if (event.key == K_1):
                for variablenode in variablenodes:
                    variablenode.sendAllMessages()
                for factornode in factornodes:
                    factornode.sendAllMessages()
                for variablenode in variablenodes:
                    variablenode.updatemuSigma()
                count += 1
            if (event.key == K_s):
                myfile = "2dmap" + str(count) + ".png"
                pygame.image.save(screen, myfile)
                print ("Saving image to", myfile)


    updateDisplay()




cflag = 0
fflag = 0
hlflag = 1


while(1):



    Eventlist = pygame.event.get()


    for variablenode in variablenodes:

        variablenode.sendAllMessages()
        # This is all very inefficient with a lot of repeated work!
        for edge in variablenode.edges:
            edge.factor.sendAllMessages()
            for edge2 in edge.factor.edges:
                edge2.variable.updatemuSigma()
                #time.sleep(0.3)


    count += 1
    print ("All variables then factors have sent", count, "messages.")



    # event handling
    for event in Eventlist:
        if (event.type == KEYDOWN):
            if (event.key == K_h):
                MEASUREMENTsigma += 0.1
                for factornode in factornodes:
                    factornode.setLambda(MEASUREMENTsigma)
                    factornode.recalculateetaLambdaprime()
            if (event.key == K_l):
                MEASUREMENTsigma -= 0.1
                for factornode in factornodes:
                    factornode.setLambda(MEASUREMENTsigma)
                    factornode.recalculateetaLambdaprime()
            if (event.key == K_s):
                myfile = "screendump.png"
                pygame.image.save(screen, myfile)
                print ("Saving image to", myfile)



    updateDisplay()
