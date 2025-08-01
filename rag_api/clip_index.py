import os, glob, tqdm, requests, pinecone, torch
from PIL import Image
from clip_anytorch import clip
from rag_api.pinecone_utils import index

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_image(path):
    img = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    return emb.cpu().numpy().astype("float32")[0]

def ingest(folder):
    files = glob.glob(os.path.join(folder, "*"))
    batch = []
    for p in tqdm.tqdm(files):
        vec = embed_image(p)
        batch.append((os.path.basename(p), vec))
        if len(batch) == 128:
            index.upsert([(k, v) for k, v in batch])
            batch.clear()
    if batch:
        index.upsert([(k, v) for k, v in batch])

if __name__ == "__main__":
    ingest("images")