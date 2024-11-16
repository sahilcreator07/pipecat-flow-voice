import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve node_modules and public files
app.mount("/node_modules", StaticFiles(directory="frontend/node_modules"), name="node_modules")
app.mount("/static", StaticFiles(directory="frontend/public"), name="static")


@app.get("/")
async def read_root():
    return FileResponse("frontend/public/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
