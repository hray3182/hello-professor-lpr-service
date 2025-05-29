import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Import necessary components from core_setup, but not the models directly for checking here
from app_core.core_setup import app, logger, load_models, create_debug_directories, streams_data 
from app_core.routers import entry, exit # Import router modules
from app_core.utils.connection_manager import close_all_connections

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Lifespan for startup and shutdown
@asynccontextmanager
async def lifespan(app_ref: FastAPI):
    logger.info("Application startup: Initializing resources...")
    create_debug_directories() # Create directories on startup
    load_models()              # Load models on startup
    # The actual check for model loading will happen inside the handler that uses them.
    # Logging in core_setup.load_models() should indicate success/failure there.
    logger.info("Model loading sequence in lifespan complete.")
    yield
    logger.info("Shutting down. Closing all peer connections.")
    await close_all_connections(streams_data, logger) 
    logger.info("All peer connections closed during shutdown.")

app.router.lifespan_context = lifespan

# Include routers from the submodules
app.include_router(entry.entry_router)
app.include_router(exit.exit_router)

# Root path to serve index.html
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return FileResponse("static/index.html", media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True # For development
    ) 