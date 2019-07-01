# Using the official tensorflow serving image from docker hub as base image
FROM tensorflow/serving

# nginx to rever proxy if we use sagemaker, would not configure it for now
#RUN apt-get update && apt-get install -y --no-install-recommends nginx git
RUN apt-get update 

#RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv

RUN apt-get install -y python3.6 python3-pip

COPY requirements.txt requirements.txt

RUN python3.6 -m pip install pip --upgrade

RUN python3.6 -m pip install -r requirements.txt

# Copy our model folder to the container
COPY flask_app flask_app 

COPY docker_service_bootcamp.sh docker_service_bootcamp.sh
 
# rewrite the ENTRYPOINT in tensorflow/serving dockerfile
ENTRYPOINT ["/bin/bash", "./docker_service_bootcamp.sh"]
