import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42);

import os
import urllib.request

if not os.path.exists("names.txt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
        "names.txt",
    )

df = pd.read_csv("names.txt", header=None)
names = df[0].str.lower()

batchsize = 32
context_size = 8
X, Y = [], []

for name in names:
	context = [26] * context_size
	for c in name:
		idx = ord(c) - ord("a")
		X.append(context)
		context = context[1:] + [idx]
		Y.append(context)
	X.append(context)
	last_ctx = context[1:] + [26]
	Y.append(last_ctx)

X = torch.tensor(X)
Y = torch.tensor(Y)

N = len(names)
Ntrain = N * 8 // 10

Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xval, Yval = X[Ntrain:], Y[Ntrain:]