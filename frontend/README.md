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
    - Run `minikube start`
    - Run `minikube docker-env`
    - Run `eval $(minikube -p minikube docker-env)`
    - Run `minikube status` or `minikube config view`
    - Run `docker-compose build`
    - Run `docker build -t wasiufrontend:2.0 .`
    - Run `docker tag wasiufrontend:2.0 minikube/wasiufrontend:2.0`

    - Run `kubectl apply -f deployment.yml && kubectl apply -f service.yml`
    - Run `kubectl get po -A` or `kubectl get po,svc` or `kubectl get po,svc -A`
    - Run `kubectl get deployments`
    - Run `kubectl delete deployment wasiunet-microservices-deployment && kubectl delete service wasiufrontend-service`
    - Run `kubectl delete pod`
    - Run `kubectl delete namespace`
    - Run `docker-compose up -d` to run in background


- Option 2 (DockerFile)

    - Run `sudo docker run -p 80:8501 frontend:latest` or `docker run -d -p 80:8501 --name wasiufrontend frontend:latest` to run in background

- Option 3 (Development)

    - Run `streamlit run app.py`



**Here are the complete instructions and steps for creating a container called wasiufrontend, making the image available with minikube, and deploying it using docker-compose, kubectl, and minikube with kubernetes yaml files:**

`Create a Dockerfile in the root of your project directory with the necessary instructions for building your wasiufrontend image.`

`Build the image using the Dockerfile:`


docker build -t wasiufrontend:2.0 .

`Create a docker-compose.yml file in the root of your project directory with the following contents:`

version: '3'
services:
  wasiufrontend:
    image: wasiufrontend:2.0
    ports:
      - "80:80"

`Build the image and run it using docker-compose:`

docker-compose build
docker-compose up -d

`Start minikube:`

minikube start

`Make the image available to minikube by using the following command`

eval $(minikube -p minikube docker-env)

`Tag the image to be used with minikube`

docker tag wasiufrontend:2.0 $(minikube -p minikube ip):5000/wasiufrontend:2.0

`Push the image to minikube's internal registry`

docker push $(minikube -p minikube ip):5000/wasiufrontend:2.0

`Create a Kubernetes deployment using the image you built in step 2, by creating a deployment.yml file with the following content:`

apiVersion: apps/v1
kind: Deployment
metadata:
  name: wasiufrontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: wasiufrontend
  template:
    metadata:
      labels:
        app: wasiufrontend
    spec:
      containers:
      - name: wasiufrontend
        image: $(minikube -p minikube ip):5000/wasiufrontend:2.0
        ports:
        - containerPort: 80

`Create a Kubernetes service for the deployment by creating a service.yml file with the following content:`

apiVersion: v1
kind: Service
metadata:
  name: wasiufrontend
spec:
  selector:
    app: wasiufrontend
  ports:
  - name: http
    port: 80
    targetPort: 80
  type: NodePort

`Deploy the resources using kubectl apply:`

kubectl apply -f deployment.yml
kubectl apply -f service.yml

`Verify that the deployment and service are running:`

kubectl get deployments
kubectl get services

`Test the service by accessing it in the browser:`

minikube service wasiufrontend --url

`To delete the deployment and service use these commands`

kubectl delete -f deployment.yml
kubectl delete -f service.yml

*Note: You should have kubectl, minikube, and docker installed in your local machine.*

You can also use kubectl get pods to check the status of the pods and kubectl logs <pod-name> to check logs of the pods.

With these instructions, you should be able to build and deploy a container called wasiufrontend using docker-compose, kubectl, and minikube with yaml files, and make the image available with minikube and referencing it in the kubernetes deployment.

`Some Errors`

The error message you're seeing: "Get "https://<you_ip_address>:5000/v2/": dial tcp <you_ip_address>:5000: connect: connection refused" indicates that the connection to the registry at IP address <you_ip_address> on port 5000 is being refused.

`There are a few reasons why this might happen:`

The minikube registry addon is not enabled.
The minikube registry addon is not running.
The firewall is blocking the connection to the registry.
The registry is not running.
Here are a few things you can try:

`Make sure that the minikube registry addon is enabled:`

minikube addons enable registry or minikube start --extra-config=apiserver.registry-creds=allow_external


`Verify the minikube registry is running`

minikube addons list

`Restart minikube`

minikube stop
minikube start

`Also you can try to create a new cluster with the desired config`

minikube delete
minikube start --extra-config=apiserver.registry-creds=allow_external

`Also you can confirm minikube and kubectl version are compatible`

`If you are behind a firewall or proxy, try configuring the proxy settings in the Docker daemon.`

`If you are using a remote minikube cluster, make sure that the cluster and the machine you are running the command on are in the same network.`

If none of these steps solve the issue, please check the minikube and the machine logs to see if there are any other issues.

It's also recommended to check the version of minikube you are running and see if there are