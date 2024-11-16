from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router  # Make sure this import is correct

app = FastAPI()

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Pipecat Flow Editor API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
