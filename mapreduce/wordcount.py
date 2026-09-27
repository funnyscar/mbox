# complete works of shakespeare
# https://www.gutenberg.org/ebooks/100

import modal
import json

app = modal.App("wordcount")

image = (modal.Image
	.debian_slim()
	.pip_install("beautifulsoup4", "requests"))

@app.function()
def mapper(chunk):
	return [(c.lower(), 1) for c in chunk.split()]

@app.function()
def reducer(items):
	from collections import defaultdict

	pid, pairs = items

	res = defaultdict(int)
	for k, v in pairs:
		res[k] += v

	return pid, res


@app.function(image=image)
def master(link):
	import requests
	from bs4 import BeautifulSoup
	from collections import defaultdict

	# parse html link
	html = requests.get(link, timeout=10).text
	soup = BeautifulSoup(html, "html.parser")
	for tag in soup(["script", "style"]):     # drop non-content
		tag.decompose()
	wordbabble = soup.get_text(separator="\n")

	# chunkify
	n_chunks = 8 # number of map workers
	lines = [l for l in wordbabble.splitlines() if l.strip()]
	size = len(lines) // n_chunks + 1
	chunks = ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]

	# MAP
	map_out = list(mapper.map(chunks))

	# SHUFFLE
	R = 8
	shuffled = defaultdict(list)
	for m in map_out:
		for k, v in m:
			rid = hash(k) % R
			shuffled[rid].append((k,v))
	
	# REDUCE
	reduce_out = list(reducer.map(shuffled.items()))
	return reduce_out

@app.local_entrypoint()
def main():
	out = master.remote("https://www.gutenberg.org/cache/epub/100/pg100-images.html")
	for rid, body in out:
		with open(f"{rid}.json", "w") as f:
			json.dump(body, f)
