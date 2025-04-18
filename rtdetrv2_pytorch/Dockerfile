# tensorrt:23.01-py3 (8.5.2.2)
FROM nvcr.io/nvidia/tensorrt:23.01-py3

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["/bin/bash"]
