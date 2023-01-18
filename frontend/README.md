Steps 1 - Change directory to `frontend` folder
    
- From `WasiuNet` root folder `cd` to `frontend` using `cd frontend`

Step 2 - Install dependencies

- Option 1 (DockerFile)

    - Install docker locally
    -Build a docker image using the `Dockerfile` by running `docker build -t frontend:latest .`

- Option 2 (Development)

    - Install python
    - Install `pipenv`
    - Install requirments using `pipenv install`

Step 3 - Run project

- Option 1 (docker-compose)

    - Run `docker-compose build`
    - Run `docker-compose start` or `docker-compose up -d` to run in background

- Option 2 (DockerFile)

    - Run `sudo docker run -p 80:8501 frontend:latest` or `docker run -d -p 80:8501 --name wasiufrontend frontend:latest` to run in background

- Option 3 (Development)

    - Run `streamlit run app.py`