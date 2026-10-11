# speech2speech
# full-duplex
# audio > codec > audio tokens > (sometimes frozen) transformer > tokens > audio

# enc = rvq
# 1d convolutions > codec > vector > matched to nearest in codebook (cluster of repeated vectors)
# (learned via vae etc)

# codec 
# 1d conv layer (weights learned) kernel is over a chunk of time
# kernels overlap
# (EMB, Frames) 
# 
# 1d conv > (emb, frames) > encode transformer/lstm > embedding > rvq token > rvq embed > decode trns/lstm (separate) >unconv > audio LOSS func on both models
# 
# rvq has a fixed size with embeddings that are learned in addition to the backprop





