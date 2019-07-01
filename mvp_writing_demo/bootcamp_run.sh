# build serving image with Dockerfile based on tensorflow/serving
docker build -t bootcamp_demo .

# run container and publish container's 5000 port to the host 5000 port and bind mount bootcamp_serve_models to bootcamp_demo
docker run -p 5000:5000 --mount type=bind,source="$(pwd)"/bootcamp_serve_models,target=/models/bootcamp_demo -it bootcamp_demo
