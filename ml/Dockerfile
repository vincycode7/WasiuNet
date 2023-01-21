FROM python:3.10.9-alpine3.17
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5002
COPY . /app
CMD ["python", "app.py"]