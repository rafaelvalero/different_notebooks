# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: text39
#     language: python
#     name: text39
# ---

"""
This is just to enjoy/play this interesting library (BERTopics) and their grapsh
Inspired here: https://ml2021.medium.com/clustering-with-python-hdbscan-964fb8292ace

"""

name_notebook = 'bertopics_minimum_spanning_tree'

# +
#from ipywidgets import  IProgress
# -

# %load_ext autoreload
# %autoreload 2
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import os
import hdbscan



# For a sorter and faster version
sort_version = True
data_original = fetch_20newsgroups(subset='all',
                                   remove=('headers', 'footers', 'quotes'))
# +

if sort_version:
    nr_topics = 10
    data = {}
    number_to_select = 10000
    for key in data_original.keys():
        print(key)
        data[key] = data_original[key][:number_to_select]
else:
    nr_topics = 'auto'
    data = data_original.copy()
docs = data['data'] 
# -

# %%time
hdbscan_model = hdbscan.HDBSCAN(gen_min_span_tree=True,
                                 metric='euclidean',
                                cluster_selection_method='eom')

topic_model = BERTopic(nr_topics=nr_topics,
                       calculate_probabilities=True, 
                       verbose=True,
                       hdbscan_model=hdbscan_model)
topics, probs = topic_model.fit_transform(docs)

hdbscan_model.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=8,
                                      edge_linewidth=2)
"""
hdbscan_model.fit(data)
hdbscan_model.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=8,
                                      edge_linewidth=2)
"""

# to save
#os.system("jupytext --output {}.py {}.ipynb".format(name_notebook,name_notebook))


