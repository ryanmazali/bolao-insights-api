from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from api.routes.admin import router as admin_router
from api.routes.player_metrics import load_player_metrics, router as player_metrics_router
from api.routes.predict import load_models, router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    load_player_metrics()
    yield


app = FastAPI(
    title="Bolão Insights API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(player_metrics_router)
app.include_router(admin_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "Bolão Insights API"}


@app.get("/health")
def health():
    return {"status": "healthy"}
