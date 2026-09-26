import modal


image = modal.Image.debian_slim().pip_install("fastapi")
app = modal.App("bb_app", image=image)

@app.function() #flags this func to modal
@modal.asgi_app() # Asynchronous Server Gateway Interface
def api():
	from fastapi import FastAPI
	web = FastAPI()

	@web.get("/")
	def root():
		return {"status": "ok"}

	@web.post("/status")
	def echo():
		return {"status": "bad"}
	
	return web


# OR SINGLE ENDPOINT

# @app.function()
# @modal.fastapi_endpoint(method="GET")
# def hello(name: str = "world"):
#     return {"message": f"Hello, {name}!"}