FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL name="instantmesh" maintainer="instantmesh"

# Add a volume for downloaded models
VOLUME /workspace/models

# create workspace folder and set it as working directory
RUN mkdir -p /workspace/instantmesh
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# create and activate conda environment
RUN conda create -n instant-generator-latest python=3.10 -y && \
    echo "source activate instant-generator-latest" > ~/.bashrc && \
    conda clean -afy
ENV PATH=/workspace/miniconda3/envs/instant-generator-latest/bin:$PATH

RUN conda run -n instant-generator-latest conda install Ninja -y
RUN conda run -n instant-generator-latest conda install cuda -c nvidia/label/cuda-12.4.1 -y
RUN conda run -n instant-generator-latest pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n instant-generator-latest pip install --no-cache-dir xformers==0.0.22.post7
RUN conda run -n instant-generator-latest pip install --no-cache-dir triton

# change the working directory to the repository
WORKDIR /workspace/instantmesh

# other dependencies
ADD ./requirements-generated.txt /workspace/instantmesh/requirements-generated.txt
RUN conda run -n instant-generator-latest pip install --no-cache-dir -r requirements-generated.txt

COPY . /workspace/instantmesh

# Run the command when the container starts
CMD ["conda", "run", "--no-capture-output", "-n", "instant-generator-latest", "uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8001"]