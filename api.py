from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from reranker import HybridReranker  # Import your HybridReranker class

# Initialize FastAPI app
app = FastAPI()

# Initialize the HybridReranker
reranker = HybridReranker()

# Request model for input validation
class RerankRequest(BaseModel):
    query: str
    candidates: list[str]
    top_n: int = 3  # Optional; defaults to 3

# Response model
class RerankResponse(BaseModel):
    ranked_candidates: list[str]

# Function to verify API Key
def verify_api_key(api_key: str = Header(...)):
    valid_api_key = "your-secret-api-key"  # Store this securely
    if api_key != valid_api_key:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# Define an endpoint for reranking with API Key authentication
@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest, api_key: str = Depends(verify_api_key)):
    """
    Rerank endpoint for the HybridReranker, protected by API Key.
    """
    try:
        ranked_candidates = reranker.rerank(
            query=request.query,
            candidates=request.candidates
        )[:request.top_n]
        return {"ranked_candidates": ranked_candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
