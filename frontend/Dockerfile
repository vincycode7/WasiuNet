FROM python:3.10.9-slim
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8501
RUN apt-get update && apt-get install -y curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/healthz || exit 1

CMD streamlit run app.py
# EXPOSE 8501
# HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=5 \
#             CMD curl -f http://localhost:8501/healthz || exit 1
# CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]