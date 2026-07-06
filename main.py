from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="Background Remover"
)

app.include_router(router)