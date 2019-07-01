docker build -t bootcamp_demo .

docker run -p 5000:5000 --mount type=bind,source="$(pwd)"/bootcamp_serve_models,target=/models/bootcamp_demo -it bootcamp_demo
