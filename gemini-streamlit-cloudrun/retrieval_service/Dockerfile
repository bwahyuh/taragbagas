# retrieval_service/Dockerfile

FROM python:3.10-slim

WORKDIR /app
COPY retrieval_service/ ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "run_app.py"]