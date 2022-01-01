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
Trying to use PyLDAvis visualization for topics modeling.

"""

name_notebook = 'bertopics_pyldavis'

# +
#from ipywidgets import  IProgress
# -

# %load_ext autoreload
# %autoreload 2
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import os
import pyLDAvis

# For a sorter and faster version
sort_version = True



data_original = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))


# +

if sort_version:
    nr_topics = 20
    data = {}
    number_to_select = 5000
    for key in data_original.keys():
        print(key)
        data[key] = data_original[key][:number_to_select]
else:
    nr_topics = 'auto'
    data = data_original.copy()
docs = data['data'] 
# -

# %%time
topic_model = BERTopic(nr_topics = nr_topics, 
                       calculate_probabilities=True, 
                       verbose = True)
topics, probs = topic_model.fit_transform(docs)

# # Create function to use visualization as in PyLDAVis
# From https://github.com/MaartenGr/BERTopic/issues/196

# +
# Prepare data for PyLDAVis
top_n = 5
R = 10

topic_term_dists = topic_model.c_tf_idf.toarray()[:top_n+1, ]
new_probs = probs[:, :top_n]
outlier = np.array(1 - new_probs.sum(axis=1)).reshape(-1, 1)
doc_topic_dists = np.hstack((new_probs, outlier))
doc_lengths = [len(doc) for doc in docs]
vocab = [word for word in topic_model.vectorizer_model.vocabulary_.keys()]
term_frequency = [topic_model.vectorizer_model.vocabulary_[word] for word in vocab]

data = {'topic_term_dists': topic_term_dists,
        'doc_topic_dists': doc_topic_dists,
        'doc_lengths': doc_lengths,
        'vocab': vocab,
        'term_frequency': term_frequency}

# Visualize using pyLDAvis
vis_data= pyLDAvis.prepare(**data,R=R, n_jobs = 1, mds='mmds')
pyLDAvis.display(vis_data)
# -

pyLDAvis.save_html(vis_data,"pyldavis.html")



# to save
os.system("jupytext --output {}.py {}.ipynb".format(name_notebook,name_notebook))


