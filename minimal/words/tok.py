
from collections import Counter
import json



num_merges = 50
vocab = {i: bytes([i]) for i in range(256)}
merges = {}

def merge(w, pair, new_id):
    out, i = [], 0
    while i < len(w):
        if i < len(w) - 1 and (w[i], w[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(w[i]); i += 1
    return tuple(out)



def read_saved():
	with open("tok.json", r) as f:
		data = json.load(f)

		merges = {(a, b): i for a, b, i in data["merges"]}
		vocab = {i: bytes([i]) for i in range(256)}

		for (a, b), i in merges.items():
			vocab[i] = vocab[a] + vocab[b]

def encode(word):
    byte_enc = tuple(word.encode())
    while True:
        # if theres a merge use it
        pairs = [p for p in zip(byte_enc, byte_enc[1:]) if p in merges]
        if not pairs:
            return list(byte_enc)

		# keep going until theres no more
        best = min(pairs, key=merges.get) # lower index ^^frequency
        byte_enc = merge(byte_enc, best, merges[best])

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode()


if __name__ == "__main__":

	with open("shakespeare.txt", "r", encoding="utf-8") as f:
		text = f.read()

	words = Counter(tuple(w.encode()) for w in text.split())

	
	for new_id in range(256, 256 + num_merges):
		pairs = Counter()

		# all words found so far
		for w, freq in words.items():
			# pairing up neighboring chars in a word
			# (116, 104, 101) -> (116,104),(104,101)
			for p in zip(w, w[1:]):
				pairs[p] += freq
		if not pairs:
			break

		# most common (x, x) pair in all words
		best = max(pairs, key=pairs.get)


		# most common substr
		merges[best] = new_id
		vocab[new_id] = vocab[best[0]] + vocab[best[1]]

		# new redesign the text
		words = Counter({merge(w, best, new_id): f for w, f in words.items()})


	with open("tok.json", "w") as f:
		json.dump({"merges": [[a, b, i] for (a, b), i in merges.items()]}, f)