FROM python:3.10.9-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8501/healthz || 
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# RUN apt-get update && apt-get install -y curl
# HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
#   CMD curl -f http://localhost:8501/healthz || exit 1