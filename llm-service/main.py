from fastapi import FastAPI

app = FastAPI(title="AI Agent API")

@app.get("/")
def read_root():
    return {"message": "LLM Service is running."}