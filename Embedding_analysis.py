# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:40:57 2024

@author: alber
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import umap


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


################2D embedding to visualize data
#t-SNE

X_reduced = PCA(n_components=100).fit_transform(X)
X_embedded = TSNE(n_components=2).fit_transform(X_reduced)


palette = sns.diverging_palette(145, 300, n=2)

fig, axs = plt.subplots(1, 3,figsize=(16,5))


#Plots per essay
ax = axs[0]
ax.axis('equal')
# palette_essays = sns.diverging_palette(145, 300, n=126)

palette_essays = sns.color_palette("hls", 126)
# plt.figure(figsize = (6,6))
for k,i in enumerate(np.unique(essay_dataset["essay_id"])):
    essay_indices = essay_dataset[essay_dataset["essay_id"] == i].index.to_list()
    ax.plot(X_embedded[essay_indices][:,0], X_embedded[essay_indices][:,1],'o', markersize = 3, color=palette_essays[k])
ax.set_title('Sentences by essay', fontsize=18)
plt.show()
# plt.savefig("Visualization2D_Essay.pdf")


#Plots per theme
ax = axs[1]
ax.axis('equal')
with open('Sentences_by_theme.json', 'r') as file:
    themes = json.load(file)

# plt.figure(figsize = (6,6))

for n,i in enumerate(list(themes.values())):
    ax.plot(X_embedded[i][:,0], X_embedded[i][:,1],'o', markersize = 4,color=palette[n] )

for i in range(len(essay_dataset)):
    if i not in themes['Employment and earnings'] and i not in themes['Education and learning']:
        ax.plot(X_embedded[i,0], X_embedded[i,1],'o', markersize = 3,color='lightgray',zorder=0)

ax.legend(['Employment and earnings', 'Education and learning', 'Other'], fontsize=15)
ax.set_title('Sentences by theme', fontsize=18)
# plt.savefig("Visualization2D_Themes2.pdf")

#Plots human vs GPT

ax = axs[2]
ax.axis('equal')
# plt.figure(figsize = (6,6))
ax.plot(X_embedded[human_indices][:,0], X_embedded[human_indices][:,1],'o', markersize = 3, color=palette[0])
ax.plot(X_embedded[gpt_indices][:,0], X_embedded[gpt_indices][:,1],'o', markersize = 3, color=palette[1])
ax.legend(['Human-written', 'GPT-written'], loc = 4, fontsize=15)
ax.set_title('Sentences by author', fontsize=18)
# plt.show()
# plt.savefig("Visualization2D_Author2.pdf")
fig.tight_layout()
plt.show()
plt.savefig("Visualization2D_All2.pdf")


###############
#UMAP
reducer = umap.UMAP(n_components=2)
UMAP_embedding = reducer.fit_transform(X)

#Plots human vs GPT
plt.figure(figsize = (6,6))
plt.plot(UMAP_embedding[human_indices][:,0], UMAP_embedding[human_indices][:,1],'o', markersize = 3, color=palette[0])
plt.plot(UMAP_embedding[gpt_indices][:,0], UMAP_embedding[gpt_indices][:,1],'o', markersize = 3, color=palette[1])
plt.legend(['Human-written', 'GPT-written'])
# plt.savefig("Visualization2D_Author.pdf")
plt.show()

#Plots per theme
with open('Sentences_by_theme.json', 'r') as file:
    themes = json.load(file)

plt.figure(figsize = (6,6))
for n,i in enumerate(list(themes.values())):
    plt.plot(UMAP_embedding[i][:,0], UMAP_embedding[i][:,1],'o', markersize = 4,color=palette[n] )

for i in range(len(essay_dataset)):
    if i not in themes['Employment and earnings'] and i not in themes['Education and learning']:
        plt.plot(UMAP_embedding[i,0], UMAP_embedding[i,1],'o', markersize = 3,color='lightgray',zorder=0)

plt.legend(['Employment and earnings', 'Education and learning', 'Other'])
# plt.savefig("Visualization2D_Themes.pdf")
plt.show()

#Plots per essay

palette_essays = sns.color_palette("hls", 126)
plt.figure(figsize = (6,6))
for k,i in enumerate(np.unique(essay_dataset["essay_id"])):
    essay_indices = essay_dataset[essay_dataset["essay_id"] == i].index.to_list()
    plt.plot(UMAP_embedding[essay_indices][:,0], UMAP_embedding[essay_indices][:,1],'o', markersize = 3, color=palette_essays[k])
plt.title('Sentences by essay')
plt.show()




