import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from modularity_maximization import partition
from modularity_maximization.utils import get_modularity
import scipy.sparse.linalg


#### Use for Karate Club Data###
data = nx.read_gml('/home/himanshu/sma_data/karate.gml', label='id')

#### Use for Dolphins Data###
#data = nx.read_gml('/home/himanshu/sma_data/dolphins.gml', label='id')

#### Use for Jazz Data###
#data = nx.read_gml('/home/himanshu/sma_data/jazz.gml', label='id')

modularity_max = data
comm_dict = partition(modularity_max)

for comm in set(comm_dict.values()):
    min_max_communities = []
    print("Community %d"%comm)
    # print(', '.join([node for node in comm_dict if comm_dict[node] == comm]))
    min_max_communities.append( [node for node in comm_dict if comm_dict[node] == comm])
    print(min_max_communities)

print('modularity-based clustering (modularity maximization) for given data is %.3f' % get_modularity(modularity_max, comm_dict))