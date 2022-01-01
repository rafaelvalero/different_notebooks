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
I try to reproduce this https://www.youtube.com/watch?v=pWrmlaXKnB8
https://www.datacamp.com/community/tutorials/introduction-factor-analysis
"""

name_notebook = 'adding_omega'

import os
import pandas as pd
import numpy as np
"""conda install -c desilinguist factor_analyzer"""
from factor_analyzer import FactorAnalyzer

# +

"""data from
http://afhayes.com/spss-sas-and-r-macros-and-code.html 
In Section OMEGA
"""
data = pd.read_csv("blirt8.csv",header = None)
data.head()
# -



# +
fa = FactorAnalyzer(n_factors=1,rotation='oblimin',method='ml')
fa.fit(data)
"""Get loading
https://factor-analyzer.readthedocs.io/en/latest/factor_analyzer.html#module-factor_analyzer.factor_analyzer"""
print(fa.loadings_)
print(fa.get_communalities())

sum_loadings_square = sum(fa.loadings_)**2
omega = sum_loadings_square / (sum_loadings_square + (1-fa.get_communalities()).sum())
print('Omega estimation:', omega)


# -

"""Wrapping omega in a function"""
def omega_mcdonalds_estimation(data, n_factors =1, 
                     rotation ='oblimin', 
                     method ='ml'):
    """
    This function estimate Omega MacDonals, as https://personality-project.org/r/psych/HowTo/omega.tutorial/omega.html
    :param data: pandas data frame.
    :param n_factors: Number of factors to used.
    :param rotation:
    :param method:
    :return: omega. float. 
    """
    fa = FactorAnalyzer(n_factors=n_factors,
                        rotation=rotation,
                        method=method)
    fa.fit(data)
    # https://en.wikipedia.org/wiki/Factor_analysis#Definition
    sum_loadings_square = sum(fa.loadings_) ** 2
    # https://personality-project.org/r/psych/HowTo/omega.tutorial/omega.html
    omega = sum_loadings_square / (sum_loadings_square + (1 - fa.get_communalities()).sum())
    return omega[0]



omega_mcdonalds_estimation(data)

# to save
os.system("jupytext --output {}.py {}.ipynb".format(name_notebook,name_notebook))


