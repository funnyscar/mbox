import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
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
context_size = 3
X, Y = [], []

for name in names:
	context = [0] * context_size
	for c in name:
		idx = ord(c) - ord("a")
		X.append(context)
		Y.append(idx)
		context = context[1:] + [idx]
	X.append(context)
	Y.append(26)

X = torch.tensor(X)
Y = torch.tensor(Y)

N = len(names)
Ntrain = N * 8 // 10

Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xval, Yval = X[Ntrain:], Y[Ntrain:]


