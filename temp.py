from fastapi import FastAPI
from app.api import recognition

app = FastAPI()

app.include_router(recognition.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)