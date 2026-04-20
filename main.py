import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# --- Routers are uncommented as steps 4–7 are implemented ---
from routers import ngram
from routers import rnn
from routers import embeddings
from routers import llm_era

app = FastAPI(title="LLM Evolution Explainer")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb_str = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logging.error(f"Unhandled exception on {request.url}:\n{''.join(tb_str)}")
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.on_event("startup")
async def startup_event():
    rnn.start_pretraining()
    embeddings.start_loading()
    llm_era.start_loading()


BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- Register routers ---
app.include_router(ngram.router, prefix="/api/ngram", tags=["ngram"])
app.include_router(rnn.router, prefix="/api/rnn", tags=["rnn"])
app.include_router(embeddings.router, prefix="/api/embeddings", tags=["embeddings"])
app.include_router(llm_era.router, prefix="/api/llm-era", tags=["llm-era"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/module/1", response_class=HTMLResponse)
async def module_ngram(request: Request):
    return templates.TemplateResponse("module-1-ngram.html", {"request": request})


@app.get("/module/1b", response_class=HTMLResponse)
async def module_training(request: Request):
    return templates.TemplateResponse("module-1b-training.html", {"request": request})


@app.get("/module/1c", response_class=HTMLResponse)
async def module_neuron(request: Request):
    return templates.TemplateResponse("module-1c-neuron.html", {"request": request})


@app.get("/module/2", response_class=HTMLResponse)
async def module_embeddings(request: Request):
    return templates.TemplateResponse("module-2-embeddings.html", {"request": request})


@app.get("/module/3", response_class=HTMLResponse)
async def module_rnn(request: Request):
    return templates.TemplateResponse("module-3-rnn-lstm.html", {"request": request})


@app.get("/module/4", response_class=HTMLResponse)
async def module_llm(request: Request):
    return templates.TemplateResponse("module-4-llm-era.html", {"request": request})


@app.get("/module/4b", response_class=HTMLResponse)
async def module_hallucinations(request: Request):
    return templates.TemplateResponse("module-4b-hallucinations.html", {"request": request})


@app.get("/module/5", response_class=HTMLResponse)
async def module_compare(request: Request):
    return templates.TemplateResponse("module-5-compare.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok"}
