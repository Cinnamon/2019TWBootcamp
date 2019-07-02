# MVP Upload Demo

This is a demo for Cinnamon 2019 Bootcamp MVP. 

# Getting Started
### Prerequisites:

#### Clone the repository:
First, please clone the repository.
```shell
clone https://github.com/Cinnamon/2019TWBootcamp/tree/master/mvp_upload_demo

cd mvp_upload_demo
```

#### Download weight
Before start building the application, you need to download the [weight](https://drive.google.com/open?id=1eM-pYWbXR3JxqqQ2PqrBgeAs4vAGcE5p) first.

The working tree should look like below.
```
mvp_upload_demo
+-- bootcamp_run.sh
+-- docker_service_bootcamp.sh
+-- Dockerfile
+-- requirements.txt
+-- README.md
+-- bootcamp_serve_models
|   +-- 1
|       +-- saved_model.pb
|       +-- variables
|           +-- variables.data-00000-of-00001
|           +-- variables.index
+-- flask_app
|   +-- app_bootcamp2.py
|   +-- data.json
|   +-- static
|   +-- templates
|   +-- uploads
```

#### Install Docker
You also need to install docker for running this demo. You can follow the official documents for installing instructions.

  * [ Install Docker for MacOS](https://docs.docker.com/docker-for-mac/install/)
  * [Install Docker for Windows](https://docs.docker.com/docker-for-windows/install/)

All other dependencies are all listed in requirements.txt file, and they will be installed in Docker container. You don't need to worry about them here.


### Running:
To run the demo, you just need to run **bootcamp_run.sh** script file.
```shell
bash bootcamp_run.sh
```

Inside **bootcamp_run.sh** script file, it will first create a new Docker Image and run a Docker container based on that Image.
```shell
# build serving image with Dockerfile based on tensorflow/serving
docker build -t bootcamp_demo2 .

# run container and publish container's 5000 port to the host 5000 port and bind mount bootcamp_serve_models to bootcamp_demo2
docker run -p 5000:5000 --mount type=bind,source="$(pwd)"/bootcamp_serve_models,target=/models/bootcamp_demo2 -it bootcamp_demo2
```

# Acknowledgements

* [Docker](https://docs.docker.com/)
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
* [keras-flask-deploy-webapp](https://github.com/mtobeiyf/keras-flask-deploy-webapp)