# 3D Model Generation

This repository represents MIRALab’s contribution to the 3D generation component of the ENIGMA project, which focuses on the digital reconstruction of fragmented and damaged Cultural Heritage (CH) objects. Within ENIGMA, our objective is to support the transformation of incomplete visual evidence into usable three-dimensional digital assets that can assist research, restoration analysis, documentation, and immersive visualization workflows. Cultural heritage artifacts such as vases, busts, tablets, steles, and other transportable objects are frequently preserved in partial or degraded states. This creates a fundamental challenge on how to generate plausible and structurally coherent 3D representations from limited observations, often derived from a single image. 

This repository makes it possible to generate 3D meshs from single-view images through an API-driven service build by MIRALab. Our development allows for deployment stability and integration. The system packages upstream generative 3D technology into a containerized service that exposes a stable HTTP endpoint, enabling diverse applications to programmatically request 3D reconstructions and receive ready-to-use mesh outputs in standard formats. 

The generation backbone is based on InstantMesh and its associated multi-view reconstruction pipeline, which internally leverages sparse-view large reconstruction models and Zero123++ for view synthesis. MIRALab’s contribution is integration, structuring, and operationalization of these models into a production-oriented pipeline. This includes controlled model loading, dependency management, output standardization, GPU-compatible containerization, and binary response handling suitable for downstream applications. The result is a service capable of transforming a single RGB input image into structured 3D assets such as textured OBJ packages or GLB files.

Within the broader ENIGMA pipeline, this API functions as the 3D generation layer. It can be invoked after preprocessing, damage analysis, or image preparation stages, and produces 3D meshes that may later be refined, visualized in XR environments, or used for comparative analysis. This design supports reproducibility, modular integration, and scalability across institutional deployments.

API-first packaging of **InstantMesh** for MIRALAb workflows.

This repository focuses on making single-image 3D generation operational as a service (FastAPI + Docker), so MIRALAb applications can call a stable HTTP endpoint and receive ready-to-use 3D outputs.

## MIRALab's Goal

- expose 3D generation through an API endpoint,
- package runtime dependencies for reproducible execution,
- organize model loading and output handling for service usage,
- support direct binary download responses (`.zip` for OBJ assets or `.glb`).

Main service entrypoint:

- `run.py` → FastAPI app with `/generate_3d_model/`

## Repository structure (high level)

- `run.py`: API server and model generation pipeline wrapper
- `app.py`: Gradio demo-oriented script (upstream-style interactive flow)
- `configs/`: model/runtime configurations
- `src/`: core model and utility modules
- `zero123plus/`: custom pipeline components
- `Dockerfile`: GPU container setup for serving

## API contract

### Endpoint

- `POST /generate_3d_model/`

### Input

- `file`: uploaded image (multipart file)
- `format` (optional): `obj` (default) or `glb`

### Output

- If `format=obj`: ZIP file containing generated assets (`.obj`, `.mtl`, texture `.png`, optional `.mp4`, and multiview image)
- If `format=glb`: binary `.glb` file

## Quick start (local)

> Recommended: Linux + NVIDIA GPU + CUDA-compatible PyTorch.

1. Create environment (Python 3.10 recommended).
2. Install dependencies:

```bash
pip install -r requirements-generated.txt
```

3. Ensure required model assets/checkpoints are available under `models/` (as expected by `run.py`).
4. Start API:

```bash
uvicorn run:app --host 0.0.0.0 --port 8001
```

## Quick start (Docker)

Build:

```bash
docker build -t miralab-instantmesh-api .
```

Run (GPU):

```bash
docker run --gpus all -p 8001:8001 -v $(pwd)/models:/workspace/models miralab-instantmesh-api
```

The container command starts:

```bash
uvicorn run:app --host 0.0.0.0 --port 8001
```

## Example API calls

Generate OBJ package:

```bash
curl -X POST "http://localhost:8001/generate_3d_model/?format=obj" \
  -H "accept: */*" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@examples/sample.png" \
  --output generated_model.zip
```

Generate GLB:

```bash
curl -X POST "http://localhost:8001/generate_3d_model/?format=glb" \
  -H "accept: */*" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@examples/sample.png" \
  --output generated_model.glb
```

## Notes for GitHub sharing

- Large model files and generated outputs should stay out of Git history (already covered by `.gitignore` for `outputs/`, caches, etc.).
- If you need reproducibility in another environment, document exact checkpoint snapshot IDs used in `models/`.
- Keep attribution to upstream projects visible in any derivative deployment/documentation.

## Acknowledgement

This project **uses InstantMesh from TencentARC as upstream technology**.

- Upstream project: https://github.com/TencentARC/InstantMesh
- Paper: *InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models* (arXiv:2404.07191)
- Additional generation backbone used by upstream flow: Zero123++ (`sudo-ai/zero123plus-v1.2`)

## License and ownership

- This repository includes MIRALAb integration/deployment work.
- Core generation methodology and several model components originate from upstream projects (notably TencentARC/InstantMesh and Zero123++).
- Review and comply with all upstream licenses and terms for model usage/distribution.

---

Developed by **MIRALAb** for API-first 3D model generation workflows.