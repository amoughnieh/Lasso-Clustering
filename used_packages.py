import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from group_lasso import GroupLasso
from sklearn.metrics import get_scorer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from collections import defaultdict
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)
#plt.style.use('default')
from matplotlib import cm
from sklearn.cluster import MeanShift, estimate_bandwidth
import re
import json
import optuna
from sklearn.preprocessing import OneHotEncoder
import sys


import warnings
warnings.filterwarnings('ignore')