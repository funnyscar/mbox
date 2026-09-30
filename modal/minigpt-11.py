import torch
import torch.nn as nn

# https://www.tensorflow.org/datasets/catalog/tedlium

# start worrying about ddp, fsdp

d = tfds.load(name='tiny_shakespeare')['train']
d = d.map(lambda x: tf.strings.unicode_split(x['text'], 'UTF-8'))
# train split includes vocabulary for other splits
vocabulary = sorted(set(next(iter(d)).numpy()))
d = d.map(lambda x: {'cur_char': x[:-1], 'next_char': x[1:]})
d = d.unbatch()
seq_len = 100
batch_size = 2
d = d.batch(seq_len)
d = d.batch(batch_size)

class AttentionHead(nn.Module):
	def __init__(self, C, d):
		super().__init__()
		self.B, self.T, self.C, self.d = batchsize, context_size, C, d
		self.Q, self.K, self.V = nn.Linear(self.C, self.d), nn.Linear(self.C, self.d), nn.Linear(self.C, self.d)
		self.register_buffer("mask", torch.tril(torch.ones(self.T, self.T, dtype=torch.bool))) # move with the weights

	def forward(self, x):
		q, k, v = self.Q(x), self.K(x), self.V(x)
		scores = q @ k.mT / self.d**0.5
		scores = scores.masked_fill(~self.mask, float('-inf'))
		att = torch.softmax(scores, dim=-1)
		return att @ v

class Block(nn.Module):
	def __init__(self, C, d, nheads):
		super().__init__()

		self.C, self.d = C, d

		self.heads = nn.ModuleList([AttentionHead(self.C, self.d) for _ in range(nheads)])

		self.W0 = nn.Linear(self.C, self.C)

		self.mlp = nn.Sequential(
			nn.Linear(self.C, self.C*4),
			nn.ReLU(),
			nn.Linear(self.C*4, self.C),
			nn.Dropout(0.1)
		)
		self.ln1 = nn.LayerNorm(self.C)
		self.ln2 = nn.LayerNorm(self.C)

	def forward(self, x):
		h = self.ln1(x)
		all_heads = torch.cat([attn(h) for attn in self.heads], dim=-1)
		x = x + self.W0(all_heads)
		x = x + self.mlp(self.ln2(x))
		return x


class MiniGPT(nn.Module):
	def __init__(self, embedding_size=64, nheads=4, nblocks=4):
		super().__init__()

		self.B, self.T, self.C = batchsize, context_size, embedding_size
		self.d = self.C // nheads

		self.embeddings = nn.Embedding(27, self.C)
		self.pos = nn.Embedding(self.T, self.C) 

		self.blocks = nn.ModuleList([Block(self.C, self.d, nheads) for _ in range(nblocks)])

		self.ln_f = nn.LayerNorm(self.C)
		self.head = nn.Linear(self.C, 27)
	
	def forward(self, tokens):
		tok_emb, pos_emb = self.embeddings(tokens), self.pos(torch.arange(self.T))
		x = tok_emb + pos_emb            # (B, T, C)

		for block in self.blocks:
			x = block(x)

		logits = self.head(self.ln_f(x)) 
		return logits


# upload weights if available


mymodel = MiniGPT()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters()) 

lossi = []

for i in range(10000):
	ixs = torch.randint(0, Xtrain.shape[0], (batchsize,))
	ctx, target = Xtrain[ixs], Ytrain[ixs]

	logits = mymodel(ctx)

	loss = criterion(logits.mT, target)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	lossi.append(loss.item())

