from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
	import uvicorn
	print("Servidor iniciado en http://127.0.0.1:8000")
	uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)