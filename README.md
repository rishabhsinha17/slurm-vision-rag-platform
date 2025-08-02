# SLURM Vision RAG Platform

Reference implementation of a vision-grounded retrieval-augmented generation (RAG) stack that can be reproduced on a small SLURM GPU cluster and deployed behind a FastAPI service.

* **Model fine-tuning** * LLaVA-1.5-7B with LoRA adapters  
  * 4 × A100 80 GB on one SLURM node  
  * Gradient checkpointing keeps memory under 80 GB per GPU

* **Retrieval layer** * CLIP ViT-B/32 image embeddings  
  * 1 M vectors in Pinecone (cosine distance)

* **Inference API** * FastAPI running in a lightweight Docker image  
  * 50 QPS throughput at 250 ms p99 latency on a single A10G

---

## Directory layout
```
.

├── training/                 # Fine-tuning assets

│   ├── fine_tune_llava.py

│   ├── train_config.yaml

│   └── slurm_submit.sh

├── rag_api/                  # Inference service

│   ├── main.py

│   ├── clip_index.py

│   ├── pinecone_utils.py

│   └── schema.py

├── docker/                   # Container definitions

│   ├── Dockerfile_api

│   └── Dockerfile_train

├── requirements.txt

└── README.md

---
```
## Quick start

```bash
git clone [https://github.com/your-org/vision-rag.git](https://github.com/your-org/vision-rag.git)
cd vision-rag
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
````

### Fine-tune LLaVA on SLURM

Place your JSONL training set at data/train.jsonl.

Each line contains an object with a text field that concatenates image captions and prompts.
Edit training/train\_config.yaml if you want to change hyper-parameters.
Submit the job:

```bash
sbatch training/slurm_submit.sh
```

Outputs land in training/outputs/ and can be copied to object storage for later use.

### Build and push containers

```bash
# Inference image
docker build -f docker/Dockerfile_api -t vision-rag-api:latest .
# Training image (optional)
docker build -f docker/Dockerfile_train -t vision-rag-train:latest .
```

Push to your registry if needed.

### Populate the Pinecone index

```bash
export PINECONE_API_KEY=xxx
export PINECONE_ENV=us-west4-gcp
python -m rag_api.clip_index --folder /path/to/image/dataset
```

### Run the FastAPI server

```bash
export PINECONE_API_KEY=xxx
export PINECONE_ENV=us-west4-gcp
uvicorn rag_api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API reference

**POST /query**

```json
{
  "image_url": "[https://example.com/lion.jpg](https://example.com/lion.jpg)",
  "top_k": 5
}
```

**Returns**

```json
{
  "matches": [
    "lion_001.jpg",
    "lion_178.jpg",
    "lion_237.jpg",
    "lion_991.jpg",
    "lion_513.jpg"
  ]
}
```

### Environment variables

| Name             | Purpose                            |
|------------------|------------------------------------|
| PINECONE\_API\_KEY | Pinecone key                       |
| PINECONE\_ENV     | Pinecone environment string        |
| PINECONE\_INDEX   | Optional index name (default `vision-rag`) |

### Benchmark numbers

| Scenario                   | Hardware            | Throughput     | Latency p99 |
|----------------------------|---------------------|----------------|-------------|
| Fine-tune                  | 4 × A100 80 GB      | 1 epoch / hr   | –           |
| Inference FastAPI (32 req) | 1 × A10G (g5.2xlarge) | 50 QPS         | 250 ms      |
