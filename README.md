SLURM Vision RAG Platform
=========================
End‑to‑end reference implementation for a vision RAG pipeline fine‑tuned on LLaVA‑1.5‑7B and served via FastAPI.

Structure
---------
training/
  fine_tune_llava.py         PyTorch training entrypoint
  train_config.yaml          Hyper‑parameters
  slurm_submit.sh            SLURM launcher
rag_api/
  main.py                    FastAPI application
  clip_index.py              Offline embedding + indexing
  pinecone_utils.py          Pinecone helper
  schema.py                  Pydantic request / response
docker/
  Dockerfile_api             Container for inference API
  Dockerfile_train           Container for training job
requirements.txt             Python dependencies