# Dockerfile
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install required Python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY train.py train.py
COPY eval.py eval.py

CMD ["python", "src/train.py"]
