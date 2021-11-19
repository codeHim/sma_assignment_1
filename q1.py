import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import time

begin = time.time()

def to_remove_edge(g):
  d1 = nx.edge_betweenness_centrality(g)
  list_of_tuples = list(d1.items())
    
  sorted(list_of_tuples, key = lambda x:x[1], reverse = True)
  return list_of_tuples[0][0]
  
def girvan_newman(g):
    a = nx.connected_components(g)
    lena = len(list(a))
    print (' The number of connected components are ', lena)
    while (lena == 1):
        u, v = to_remove_edge(g)
        g.remove_edge(u, v) 
        
        cc = nx.connected_components(g)
        lena=len(list(a))
        print (' The number of connected components are ', lena)
        for item in cc:
            print(item)
    return a





#### Use for Karate Club Data###
data = nx.read_gml('/home/himanshu/sma_data/karate.gml', label='id')

#### Use for Dolphins Data###
#data = nx.read_gml('/home/himanshu/sma_data/dolphins.gml', label='id')

#data = nx.read_gml('/home/himanshu/sma_data/jazz.gml', label='id')


girvan = girvan_newman(data)

for item in girvan:
    print (item.nodes())

end = time.time()
print(f"Total runtime of the program is {end - begin}")

nx.draw(data, with_labels=True, font_weight='bold')
plt.show()



