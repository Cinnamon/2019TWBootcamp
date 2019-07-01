docker build -t bootcamp_demo2 .

docker run -p 5000:5000 --mount type=bind,source="$(pwd)"/bootcamp_serve_models,target=/models/bootcamp_demo2 -it bootcamp_demo2
