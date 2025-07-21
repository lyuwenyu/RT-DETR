FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["/bin/bash"]