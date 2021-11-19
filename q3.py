import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import time

begin = time.time()


#### Use for Karate Club Data###
data = nx.read_gml('/home/himanshu/sma_data/karate.gml', label='id')

#### Use for Dolphins Data###
#data = nx.read_gml('/home/himanshu/sma_data/dolphins.gml', label='id')

#### Use for Jazz Data###
#data = nx.read_gml('/home/himanshu/sma_data/jazz.gml', label='id')


print("number of nodes : ",nx.number_of_nodes(data))
print("number of edges : ",nx.number_of_edges(data))
print("average path length : ",nx.average_shortest_path_length(data))
print("average clustering coefficient : ",nx.average_clustering(data))

def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

draw_graph(data)

###adjacency matrix
W = nx.adjacency_matrix(data)
print(W.todense())


###degree matrix

D = np.diag(np.sum(np.array(W.todense()), axis=1))
print('degree matrix:')
print(D)

# laplacian matrix
L = D - W
print('laplacian matrix:')
print(L)


e, v = np.linalg.eig(L)
# eigenvalues
print('eigenvalues:')
print(e)
# eigenvectors
print('eigenvectors:')
print(v)
i = np.where(e < 0.5)[0]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(v)
  
# Normalizing the Data
X_normalized = normalize(X_scaled)
  
X_normalized = pd.DataFrame(X_normalized)
  
#Reducing dimensions
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['C1', 'C2']

spectral_model_rbf = SpectralClustering()
labels_rbf = spectral_model_rbf.fit_predict(X_principal)
end = time.time()
print(f"Total runtime of the program is {end - begin}")

colr = {}
for l in labels_rbf:
    colr[l] = 'b'


colr = ['b','y','r','g','c','m','k','w']
# Build colour vector for data points
colr_vec = [colr[label] for label in labels_rbf]

plt.figure(figsize =(9, 9))
plt.scatter(X_principal['C1'], X_principal['C2'], c = colr_vec)
plt.legend((colr_vec),('Label 0','Label 1','Label 3','Label 4','Label 5','Label 6','Label 7','Label 8'))
plt.show()



