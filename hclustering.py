#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd 
import numpy as np
import math
import json
import sys
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import combinations


# In[102]:


def read_file(file, for_analysis = False):
    df = pd.read_csv(file, header=None, sep=',')
    for column in df.columns:
        if int(df.at[df.index[0],column]) == 0:
            if for_analysis:
                df = df.set_axis(df[[column]].astype(str), axis=0)
            df.drop(column, axis=1, inplace=True)
    to_drop = df.index[0]
    shrunk_df = df.drop(to_drop)
    return shrunk_df


# In[103]:


df = read_file("data/iris.csv")
df


# ### Create Dist Matrix

# In[104]:


def euclidean_dist(point, pointArray):
    return np.sqrt(np.sum((pointArray - point) ** 2, axis=1))


# In[105]:


def calcDistMatrix(df, distFunctionVect):
    # must be fully numeric and normalized df
    dfarray = np.array(df)
    
    distMatrix = []
    for i, d in enumerate(dfarray):
        # performs Euclidean distance on all elements in data (vectorized)
        dists = distFunctionVect(dfarray[i], dfarray)
        distMatrix.append(dists)
    
    return pd.DataFrame(distMatrix)


# In[106]:


dist_matrix = calcDistMatrix(df, euclidean_dist)
dist_matrix


# ### Hcluster Setup

# In[107]:


class Leaf:
    def __init__(self, data):
        self.n_type = 'leaf'
        self.height = 0
        self.data = data
        self.str_rep = str(data)

    def __repr__(self):
        return f"type: {self.n_type}, height: {self.height}, data: {self.data}"

    def to_dict(self):
        json_dict = {}
        json_dict["type"] = self.n_type
        json_dict["height"] = self.height
        json_dict["data"] = self.data
        return json_dict

class Node:
    def __init__(self, n_type, height, nodes, str_rep):
        self.n_type = n_type
        self.height = height
        self.nodes = nodes
        self.str_rep = str_rep

    def __repr__(self):
        return f"type: {self.n_type}, height: {self.height}, nodes: {self.nodes}"

    def to_dict(self):
        json_dict = {}
        json_dict["type"] = self.n_type
        json_dict["height"] = self.height
        json_dict["nodes"] = self.nodes
        return json_dict


# In[108]:


def min_matrix(dist_matrix):
    min_locs = dist_matrix.idxmin()
    min_row = 0
    min_col = 0
    min_val = np.inf
    for val in min_locs:
        if dist_matrix.at[val, min_locs[val]] < min_val:
            min_col = val
            min_row = min_locs[val]
            min_val = dist_matrix.at[val, min_locs[val]]

    return min_row, min_col, min_val


# In[109]:


def generate_starting_clusters(dist_matrix):
    clusters = {0:[]}
    for column in dist_matrix:
        clusters[0].append(column)
    return clusters


# In[110]:


def init_dendrogram(dist_matrix):
    dgram = []
    for column in dist_matrix:
        dgram.append(Leaf(column))
    return dgram


# ### Hcluster (and dist funcs)

# In[111]:


def single_link(s, r):
    return np.minimum(s, r)


# In[112]:


def complete_link(s, r):
    return np.maximum(s, r)


# In[113]:


import copy
import re
import json
def hcluster(dist_matrix, threshold = np.inf, merge_func = single_link):
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if i == j:
                dist_matrix.iat[i,j]=np.nan #so that 0 isn't always reported as the min val

    clusters = generate_starting_clusters(dist_matrix)
    dgram = init_dendrogram(dist_matrix)
    current_height = 0
    
    while len(clusters[current_height]) > 1:
        
        s, r, min_val = min_matrix(dist_matrix) #s = row, r = column

        # Set up next level of clusters
        clusters[current_height+1] = copy.deepcopy(clusters[current_height])
        clusters[current_height+1].remove(s)
        clusters[current_height+1].remove(r)

        # Create new cluster
        new_cluster = str((s,r)).strip()
        new_cluster = re.sub('[^A-Za-z0-9,()]+','', new_cluster)

        # Insert new cluster
        clusters[current_height+1].insert(0, new_cluster)

        a = next(item for item in dgram if item.str_rep == str(s))
        b = next(item for item in dgram if item.str_rep == str(r))
        
        dgram.append(Node('node', min_val+1, [a.to_dict(), b.to_dict()], new_cluster))
        dgram.remove(a)
        dgram.remove(b)

        
        new_matrix = pd.DataFrame(np.nan, clusters[current_height+1], clusters[current_height+1]) 
        s_values = dist_matrix.loc[s]
        r_values = dist_matrix[r]
        dist_matrix.drop(index=s, inplace=True)
        dist_matrix.drop(columns=r, inplace=True)  
        for j in clusters[current_height]:
            if j != s and j != r:
                new_matrix[j] = dist_matrix[j]
        merge = merge_func(s_values, r_values)
        merge = merge.drop(s)
        merge = merge.drop(r)
        
        new_matrix[new_cluster] = merge
        for i in merge.keys():
            new_matrix.loc[new_cluster].at[i] = merge[i]      

        dist_matrix = copy.deepcopy(new_matrix)
        current_height += 1
    dgram[0].n_type = 'root'
    dgram[0].height += min_val

    dendrogram = dgram[0].to_dict()
    f = open("dendrogram.json",'w')
    json.dump(dendrogram, f, indent = 4)

    if threshold != np.inf:
        cuts = []
        cuts = cut_dgram(dendrogram, threshold)
        return create_final_clusters(cuts)
    else:
        print("Output dendrogram to")


# ### Cutting dendrogram

# In[114]:


def cut_dgram(dgram, threshold):
    s1 = []
    s2 = []

    s1.append(dgram)
    while len(s1) != 0:
        curr = s1.pop()
        
        if curr['height'] >= threshold:
            s1.append(curr['nodes'][0])
            s1.append(curr['nodes'][1])
        else:
            s2.append(curr)

    return s2


# In[115]:


def find_leaves(dgram):
    s1 = []
    s2 = []

    s1.append(dgram)
    while len(s1) != 0:
        curr = s1.pop()
        if curr['type']=='node' or curr['type'] == 'root':
            s1.append(curr['nodes'][0])
            s1.append(curr['nodes'][1])
        else:
            s2.append(curr)
     
    # Return all the leaf data
    leaves = []
    for leaf in s2:
        leaves.append(leaf['data'])
    return leaves
        


# In[116]:


def create_final_clusters(dgrams):
    cluster = 0
    clusters = {}
    for tree in dgrams:
        result = find_leaves(tree)
        clusters[cluster] = result
        cluster += 1
    return clusters


# In[117]:



dist_matrix = calcDistMatrix(df, euclidean_dist)
end_gram_single = hcluster(dist_matrix, 14)
print(end_gram_single)


# In[118]:


dist_matrix = calcDistMatrix(df, euclidean_dist)
end_gram_comp = hcluster(dist_matrix, 14, complete_link)
print(end_gram_comp)


# ### Analysis code

# In[119]:


# gets centroid of numeric dataframe (not normalized)
def calc_centroid(numdf):
    return np.divide(np.sum(np.array(numdf), axis=0),len(numdf))


# In[120]:


def calc_SSE(dfarray, c, distFunc):
    return np.sum(np.square(distFunc(c, dfarray)))


# In[121]:


def printClusterInfo(clusters, noData=False):
    for clusterInfo in clusters:
        for key in clusterInfo:
            if key == "dataPoints":
                if not noData:
                    print(f"{key}: \n{clusterInfo[key].to_markdown()}")
            else:
                print(f"{key}: {clusterInfo[key]}")
        print('\n')


# In[122]:


def analyze_clusters(df, numdf, distFunc):
    clusters=[]
    for i, c in enumerate(df['cluster'].unique()):
        info = {}
        info["clusterID"] = i
        if c is None:
            pnts = df.loc[df['cluster'].isna()]
            info["type"] = "Noise"
        else:
            pnts = df[df['cluster'] == c]
            info["type"] = "Cluster"
        
        numpnts = numdf.loc[pnts.index]
        
        info["centroid"] = calc_centroid(numpnts)
        info["SSE"] = calc_SSE(np.array(numpnts), info["centroid"], distFunc)
        
        dists = distFunc(info["centroid"], np.array(numpnts))
        df.loc[pnts.index, "distToCentroid"] = dists
        pnts = df.loc[pnts.index]
        info["maxDistToCentroid"] = max(dists)
        info["minDistToCentroid"] = min(dists)
        info["avgDistToCentroid"] = np.sum(dists)/len(pnts)
        info["numPoints"] = len(pnts)
        info["dataPoints"] = pnts
        clusters.append(info)
    return clusters


# In[123]:


def all_together(file, threshold=np.inf, dist_func=single_link, silent=False, nodata=False):
    df = read_file(file)
    dist_matrix = calcDistMatrix(df, euclidean_dist)
    end_gram = hcluster(dist_matrix, threshold, dist_func)
    dist_matrix = calcDistMatrix(df, euclidean_dist) #hcluster modifies og dist_matrix

    df = read_file(file, for_analysis=True)
    numdf = copy.deepcopy(df)

    if threshold != np.inf:
        cluster_labels=[0]*len(dist_matrix)
        for i in range(len(end_gram)):
            print()
            for idx in end_gram[i]:
                cluster_labels[idx] = i 
        df.insert(len(df.columns), 'cluster', cluster_labels)
        #print(df)
        clusters = analyze_clusters(df, numdf, euclidean_dist)
        if silent == False:
            if nodata == True:
                printClusterInfo(clusters, noData=True)
            else:
                printClusterInfo(clusters)
        return clusters, df    


# In[124]:


all_together("data/iris.csv", 3.7, complete_link)


# ### Actual Commandline Running

# In[125]:


import sys
threshold = 10 

sys.argv = f"hclustering.py ./data/mammal_milk.csv {threshold}".split(" ")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        all_together(sys.argv[1], np.inf, complete_link)
    elif len(sys.argv) == 3:
        all_together(sys.argv[1], float(sys.argv[2]), complete_link)
    else:
        print("Usage: python3 hclustering.py <Filename> [<threshold>]")
        exit(1)
