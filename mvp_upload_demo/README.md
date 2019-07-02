# MVP Upload Demo

This is a demo for Cinnamon 2019 Bootcamp MVP. 

# Getting Started
### Installing:
You need to install docker before running this demo. You can follow the official documents for installing instructions.

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