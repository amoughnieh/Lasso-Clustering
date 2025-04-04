import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from group_lasso import GroupLasso
from group_lasso.utils import extract_ohe_groups
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
import random
import os
import json
import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys


import warnings
warnings.filterwarnings('ignore')