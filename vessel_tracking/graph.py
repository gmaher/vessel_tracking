# Python program for Kruskal's algorithm to find
# Minimum Spanning Tree of a given connected,
# undirected and weighted graph

from collections import defaultdict
import numpy as np

#Class to represent a graph
class Graph:

    def __init__(self,vertices):
        self.V= vertices #No. of vertices
        self.graph = [] # default dictionary
                                # to store graph


    # function to add an edge to graph
    def addEdge(self,u,v,w):
        self.graph.append([u,v,w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else :
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
        # algorithm
    def KruskalMST(self):
        result =[] #This will store the resultant MST

        i = 0 # An index variable, used for sorted edges
        e = 0 # An index variable, used for result[]

            # Step 1:  Sort all the edges in non-decreasing
                # order of their
                # weight.  If we are not allowed to change the
                # given graph, we can create a copy of graph
        self.graph =  sorted(self.graph,key=lambda item: item[2])

        parent = [] ; rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        #while e < self.V -1 :
        for e in range(len(self.graph)):
            # Step 2: Pick the smallest edge and increment
                    # the index for next iteration
            u,v,w =  self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent ,v)

            # If including this edge does't cause cycle,
                        # include it in result and increment the index
                        # of result for next edge
            if x != y:
                #e = e + 1
                result.append([u,v,w])
                self.union(parent, rank, x, y)
            # Else discard the edge

        # print the contents of result[] to display the built MST
        return result

def DetectTree(V_free, cost_func, CD, K=20):
    Nfree = V_free.shape[0]
    edges = []
    for i in range(Nfree):
        #for each node get K nearest neighbors that are collison free
        p = V_free[i]
        dists = np.sum((V_free-p)**2,axis=1)

        dists[i] = 1e10

        idx   = np.argpartition(dists,K)[:K]

        for j in range(K):
            p_next = V_free[idx[j]]
            if not (p_next[0]==p[0] and p_next[1]==p[1]):
                if not CD.collision(p,p_next):
                    c = cost_func(p,p_next)
                    t = ((i,idx[j]), (p,p_next), c)
                    edges.append(t)

    G = Graph(Nfree)

    for e in edges:
        u = e[0][0]
        v = e[0][1]
        w = e[-1]
        G.addEdge(u,v,w)

    MST = G.KruskalMST()

    return MST,edges
