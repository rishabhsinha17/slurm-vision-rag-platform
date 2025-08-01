import time, uvicorn, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from clip_anytorch import clip
from rag_api.schema import QueryRequest, QueryResponse
from rag_api.pinecone_utils import index

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
app = FastAPI()

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    start = time.time()
    try:
        img_data = Image.open(req.image_url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    img = preprocess(img_data).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    vec = emb.cpu().numpy().astype("float32")[0]
    res = index.query(vec, top_k=req.top_k, include_metadata=False)
    matches = [m["id"] for m in res["matches"]]
    latency = (time.time() - start) * 1000
    if latency > 250:
        print(f"slow_request {latency:.1f}ms")
    return QueryResponse(matches=matches)

if __name__ == "__main__":
    uvicorn.run("rag_api.main:app", host="0.0.0.0", port=8000, workers=4)