#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 00:40:04 2019

@author: israeldiego
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
from scipy.special import gammaln
from scipy.stats import kendalltau

# The date to process
dt = "2012-04-01"

# Create a set of graphs in one pdf file.
# pdf = PdfPages("internet_%s.pdf" % dt)

# Load the traffic statistics (one record per minute
# within a day).
df = pd.read_csv("traffic_stats_%s.csv" % dt)

# Rename the columns
cname = {"Traffic": "Total traffic", "UDP": "UDP", "TCP": "TCP",
         "Sources": "Unique sources"}