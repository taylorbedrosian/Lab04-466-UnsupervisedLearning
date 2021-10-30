#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import json
import sys
import time


# In[5]:


def euclideanDist(point, pointArray):
    return np.sqrt(np.sum((pointArray - point) ** 2, axis=1))


# In[6]:


# pass a ***vectorized*** distance function: dist(point, pointArray)
def calcDistMatrix(df, distFunctionVect):
    # All but last three columns (visited, cluster, and type labels)
    dfarray = np.array(df.iloc[:,:-3])
    
    distMatrix = []
    for i, d in enumerate(dfarray):
        # performs Euclidean distance on all elements in data (vectorized)
        dists = distFunctionVect(dfarray[i], dfarray)
        distMatrix.append(dists)
    
    return pd.DataFrame(distMatrix)


# In[7]:


def calcNeighborhoods(distMatrix, epsilon):    
    # iterating through a dictionary is much faster
    dfdict = distMatrix.to_dict('records')
    
    # Sorry, had to do this in one line, filters each row by epsilon
    # k+1 as index in these datasets starts at 1
    return [[k+1 for (k,v) in row.items() if v <= epsilon] for row in dfdict]


# In[14]:


def calcCorePoints(neighborhoods, minpts):
    # > because the point itself should be excluded
    return [i+1 for i, v in enumerate(neighborhoods) if len(v) > minpts]


# In[49]:


def densityConnected(df, pntId, neighborhoods, cores, currCluster):
    # visit each unvisited neigh, and their neighbors if core. DFS 
    # update visited, clusterid, and type
    
    if df.at[pntId, "visited"]:
        return
    df.at[pntId, "visited"] = True
    for neigh in neighborhoods[pntId-1]:
        if not df.at[neigh, "visited"]:
            df.at[neigh, "cluster"] = currCluster
            if neigh in cores:
                # continue density connectivity
                df.at[neigh, "type"] = "core"
                densityConnected(df, neigh, neighborhoods, cores, currCluster)
            else:
                df.at[neigh, "visited"] = True
                df.at[neigh, "type"] = "boundary"


# In[49]:


def dbscan(df, distFunc, epsilon, minpnts):
    distMatrix = calcDistMatrix(df, euclideanDist)
    neighborhoods = calcNeighborhoods(distMatrix, epsilon)
    cores = calcCorePoints(neighborhoods, minpts)
    currCluster=0
    
    for c in cores:
        if not df.at[c, "visited"]:
            df.at[c, "type"] = "core"
            df.at[c, "cluster"] = currCluster 
            densityConnected(df, c, neighborhoods, cores, currCluster)
            currCluster += 1
    
    clusters=[]
    for c in df['cluster'].unique():
        pnts = list(df[df['cluster'] == c].index)
        info = {
            "points": pnts,
            "numPoints": len(pnts)
        }
        clusters.append(info)
    return clusters


# In[2]:


# normalizes all columns
def normalizeDf(df):
    for c in df.columns:
        colMax = df[c].max()
        colMin = df[c].min()
        
        # probably no need to normalize if the values are very small. Might have to adjust the value
#         if colMax < 1:
#             continue
        df[c] = df[c].apply(lambda x: (x - colMin)/(colMax-colMin))
    return df


# In[3]:


def readFiles(filename):
    df = pd.read_csv(datafile, header=None)
    
    # restrictions are in first row
    restr = pd.to_numeric(df.iloc[0])
    
    # drop metadata columns
    df = df.drop([0], axis=0)
    
    # cleanup numeric columns, converting from strings
    # drop columns that are specified in restriction file
    for i, v in enumerate(df.columns):
        if restr[i] < 1:
            df = df.drop(columns=[v])
        else:
            df[v] = pd.to_numeric(df[v], errors='coerce')
    
    
    # drop unknown values
    df = df.dropna()
    df = df[(df != '?').all(axis=1)]
    
    df = normalizeDf(df)
    df["visited"] = False
    df["cluster"] = None
    df["type"] = "Noise"
    return df


# In[50]:


# sys.argv = "dbscan.py ./data/iris.csv 0.4 40".split(" ")
if __name__ == "__main__":
    if len(sys.argv) == 4:
        _, datafile, epsilon, minpts = sys.argv
    else:
        print("Usage: python3 dbscan.py <datafile.csv> <epsilon> <numPoints>")
        exit(1)
    minpts = float(minpts)
    epsilon = float(epsilon)
    df = readFiles(datafile)
    
    clusters = dbscan(df, euclideanDist, epsilon, minpts)
    for c in clusters:
        for k in c:
            print(k,':',c[k])
        print()

