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

- Option 1 (DockerFile)

    - Run `sudo docker run -p 8501:8501 frontend:latest`

- Option 2 (Development)

    - Run `streamlit run app.py`