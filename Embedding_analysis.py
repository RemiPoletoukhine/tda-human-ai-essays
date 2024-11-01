# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:40:57 2024

@author: alber
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



sentence_embeddings = np.load("sentence_embeddings.npy")  #All embeddings
essay_dataset = pd.read_csv("essay_dataset.csv")


human_indices = essay_dataset[essay_dataset["author"] == "human"].index.to_list()
gpt_indices = essay_dataset[essay_dataset["author"] == "chatgpt"].index.to_list()
human_embeddings = sentence_embeddings[human_indices]
gpt_embeddings = sentence_embeddings[gpt_indices]

X = sentence_embeddings
np.array(X, dtype=np.float32)


# file = open("Topics.txt", "w")       
# for i in np.unique(essay_dataset['title']): 
#     file.write(i)
#     file.write('\n')
# file.close()

file2 = open("Topics.txt", "r")
Topics = []
lines = file2.readlines()
for line in lines: 
    a = str(line)
    Topics.append(a[:-2])
file2.close()


###################################################3
#Perform Classical clustering: DBSCAN

# db = DBSCAN(eps=1, min_samples=8).fit(X)
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print("Estimated number of clusters: %d" % n_clusters_)
# print("Estimated number of noise points: %d" % n_noise_)


#2D embedding to visualize data
X_reduced = PCA(n_components=100).fit_transform(X)
X_embedded = TSNE(n_components=2).fit_transform(X_reduced)


palette = sns.diverging_palette(145, 300, n=2)

#Plots human vs GPT
plt.figure(figsize = (6,6))
plt.plot(X_embedded[human_indices][:,0], X_embedded[human_indices][:,1],'o', markersize = 3, color=palette[0])
plt.plot(X_embedded[gpt_indices][:,0], X_embedded[gpt_indices][:,1],'o', markersize = 3, color=palette[1])
plt.legend(['Human-written', 'GPT-written'])
plt.savefig("Visualization2D_Author.pdf")
plt.show()

#Plots per essay
with open('Sentences_by_theme.json', 'r') as file:
    themes = json.load(file)

plt.figure(figsize = (6,6))
for n,i in enumerate(list(themes.values())):
    plt.plot(X_embedded[i][:,0], X_embedded[i][:,1],'o', markersize = 4,color=palette[n] )

plt.legend(['Employment and earnings', 'Education and learning'])
plt.savefig("Visualization2D_Themes.pdf")
plt.show()




