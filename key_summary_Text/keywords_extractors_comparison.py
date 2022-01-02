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



# # Aim
# The aim of this is to run a quick literature and libraries reviews and starting point for some of the most important keywords/keyphrases extractions techniques in the python universe.
#
# # Table of Contents
# 1. [Python Keyphrase Extraction](#Python Keyphrase Extraction (PKE))
# 2. [Rake](#Rake)
# 3. [KeyBERT](#KeyBERT)
# 3. [YAKE](#YAKE)
# 4. [pytextrank](#'pytextrank: TextRank Algorithm')
# 5. [References](#References)
#
#
# # References for Comparing Methods
# ### Blogs
# - Comparison popular algorithms for keyword extraction. [here](https://medium.com/mlearning-ai/10-popular-keyword-extraction-algorithms-in-natural-language-processing-8975ada5750c)
#
#
# ### Papers 
# - [Boudin, Florian. "Pke: an open source python-based keyphrase extraction toolkit." Proceedings of COLING 2016, the 26th international conference on computational linguistics: system demonstrations. 2016.](http://aclweb.org/anthology/C16-2015)
# - [Sun, Chengyu, et al. "A Review of Unsupervised Keyphrase Extraction Methods Using Within-Collection Resources." Symmetry 12.11 (2020): 1864.](https://www.mdpi.com/2073-8994/12/11/1864)
# ### Lybraries
# 1. [Python Keyphrase Extraction](https://boudinfl.github.io/pke/build/html/index.htm)
# 2. [Rake](https://pypi.org/project/rake-nltk/)
# 3. [KeyBERT](https://github.com/MaartenGr/KeyBERT)
# 3. [YAKE](https://github.com/LIAAD/yake)
#
#

# import os
# #import nltk
# #nltk.download('stopwords')
# #nltk.download('punkt')

name_notebook = 'keywords_extractors_comparison'

benchmark_text = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias)."""

# # Rake
# Documentation: https://pypi.org/project/rake-nltk/
# Examples: https://towardsdatascience.com/extracting-keyphrases-from-text-rake-and-gensim-in-python-eefd0fad582f

# +
from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()

# Extraction given the text.
r.extract_keywords_from_text(benchmark_text)

# -

# To get keyword phrases ranked highest to lowest.
r.get_ranked_phrases()

# To get keyword phrases ranked highest to lowest with scores.
r.get_ranked_phrases_with_scores()

# # KeyBERT
# +https://github.com/MaartenGr/KeyBERT
# +https://doi.org/10.5281/zenodo.4461265
# +https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea?source=user_profile---------8-------------------------------

from keybert import KeyBERT
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(benchmark_text)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(1, 1), stop_words=None)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(1, 2), stop_words=None)


keywords = kw_model.extract_keywords(benchmark_text, highlight=True)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_maxsum=True, nr_candidates=20, top_n=5)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_mmr=True, diversity=0.7)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(3, 3),
                          stop_words='english', 
                          use_mmr=True, diversity=0.2)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(3, 3),
                          stop_words='english', 
                          use_mmr=False, diversity=0.2)

# ## KeyBERT: with embeddings

from keybert import KeyBERT
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(3, 3),
                          stop_words='english', 
                          use_mmr=False, diversity=0.2)

kw_model.extract_keywords(benchmark_text, keyphrase_ngram_range=(3, 3),
                          stop_words='english', 
                          use_mmr=True, diversity=0.2)

# # YAKE
# See for the papers: https://github.com/LIAAD/yake

import yake

# +
kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords(benchmark_text)

for kw in keywords:
	print(kw)

# +
language = "en"
max_ngram_size = 3
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, 
                                            n=max_ngram_size, 
                                            dedupLim=deduplication_thresold, 
                                            dedupFunc=deduplication_algo, 
                                            windowsSize=windowSize, 
                                            top=numOfKeywords, 
                                            features=None)
keywords = custom_kw_extractor.extract_keywords(benchmark_text)

for kw in keywords:
    print(kw)

# +
from yake.highlight import TextHighlighter

th = TextHighlighter(max_ngram_size = 3)
th.highlight(benchmark_text, keywords)
# -

# # Python Keyphrase Extraction (PKE)
#
# Documentation: https://github.com/boudinfl/pke
#
# Pke currently implements the following keyphrase extraction models:
#
#
# `pke` currently implements the following keyphrase extraction models:
#
# * Unsupervised models
#   * Statistical models
#     * TfIdf [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#tfidf)]
#     * KPMiner [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#kpminer), [article by (El-Beltagy and Rafea, 2010)](http://www.aclweb.org/anthology/S10-1041.pdf)]
#     * YAKE [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#yake), [article by (Campos et al., 2020)](https://doi.org/10.1016/j.ins.2019.09.013)]
#   * Graph-based models
#     * TextRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#textrank), [article by (Mihalcea and Tarau, 2004)](http://www.aclweb.org/anthology/W04-3252.pdf)]
#     * SingleRank  [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#singlerank), [article by (Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)]
#     * TopicRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicrank), [article by (Bougouin et al., 2013)](http://aclweb.org/anthology/I13-1062.pdf)]
#     * TopicalPageRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#topicalpagerank), [article by (Sterckx et al., 2015)](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)]
#     * PositionRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#positionrank), [article by (Florescu and Caragea, 2017)](http://www.aclweb.org/anthology/P17-1102.pdf)]
#     * MultipartiteRank [[documentation](https://boudinfl.github.io/pke/build/html/unsupervised.html#multipartiterank), [article by (Boudin, 2018)](https://arxiv.org/abs/1803.08721)]
# * Supervised models
#   * Feature-based models
#     * Kea [[documentation](https://boudinfl.github.io/pke/build/html/supervised.html#kea), [article by (Witten et al., 2005)](https://www.cs.waikato.ac.nz/ml/publications/2005/chap_Witten-et-al_Windows.pdf)]
#     * WINGNUS [[documentation](https://boudinfl.github.io/pke/build/html/supervised.html#wingnus), [article by (Nguyen and Luong, 2010)](http://www.aclweb.org/anthology/S10-1035.pdf)]
#
#

# +
import pke

# initialize keyphrase extraction model, here TopicRank
extractor = pke.unsupervised.TopicRank()

# -

extractor.load_document(input=benchmark_text, 
                        language='en_core_web_sm')

# +

# keyphrase candidate selection, in the case of TopicRank: sequences of nouns
# and adjectives (i.e. `(Noun|Adj)*`)
extractor.candidate_selection()



# +
# candidate weighting, in the case of TopicRank: using a random walk algorithm
extractor.candidate_weighting()


# -

# N-best selection, keyphrases contains the 10 highest scored candidates as
# (keyphrase, score) tuples
keyphrases = extractor.get_n_best(n=10)
keyphrases

# ## TF-IDF

# +
import string
import pke

# 1. create a TfIdf extractor.
extractor = pke.unsupervised.TfIdf()

# 2. load the content of the document.
extractor.load_document(input=benchmark_text,
                        language='en_core_web_sm',
                        normalization=None)

# 3. select {1-3}-grams not containing punctuation marks as candidates.
extractor.candidate_selection(n=3, stoplist=list(string.punctuation))

# 4. weight the candidates using a `tf` x `idf`
extractor.candidate_weighting()

# 5. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)
print(keyphrases)
# -

# ## TextRank

# +
import pke

# define the set of valid Part-of-Speeches
pos = {'NOUN', 'PROPN', 'ADJ'}

# 1. create a TextRank extractor.
extractor = pke.unsupervised.TextRank()

# 2. load the content of the document.
extractor.load_document(input=benchmark_text,
                        language='en_core_web_sm',
                        normalization=None)

# 3. build the graph representation of the document and rank the words.
#    Keyphrase candidates are composed from the 33-percent
#    highest-ranked words.
extractor.candidate_weighting(window=2,
                              pos=pos,
                              top_percent=0.33)

# 4. get the 10-highest scored candidates as keyphrases
keyphrases = extractor.get_n_best(n=10)
print(keyphrases)
# -

extractor.build_word_graph(window=2, pos=None)
extractor.graph

import networkx as nx
import matplotlib.pyplot as plt
nx.draw(extractor.graph)
plt.show()

# # pytextrank: TextRank Algorithm
# https://pypi.org/project/pytextrank/
# https://derwen.ai/docs/ptr/sample/

import spacy
import pytextrank

# load a spaCy model, depending on language, scale, etc.
# python3 -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", config={ "stopwords": { "word": ["NOUN"] } })


# +
# add PyTextRank to the spaCy pipeline
doc = nlp(benchmark_text)

# examine the top-ranked phrases in the document
for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)
# -



# # Gensim Summarization (DEPRECATED)
# https://radimrehurek.com/gensim_3.8.3/summarization/keywords.html
#
# https://stackoverflow.com/a/68023653/7127519

# +
#from gensim.summarization import keywords

# +
#keywords(text).split('\n')
# -

# to save
os.system("jupytext --output {}.py {}.ipynb".format(name_notebook,name_notebook))
os.system("jupytext --output {}.md {}.ipynb".format(name_notebook,name_notebook))


