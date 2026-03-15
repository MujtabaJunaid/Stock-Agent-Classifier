from backend.main import app
import uvicorn

if __name__ == "__main__":
    # Redirect execution to the new backend structure
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)