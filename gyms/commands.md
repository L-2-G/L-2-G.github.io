# Create the Docker Image:

cd l-2-g.github.io


docker build gyms --tag l2g

# Run the Docker Image:

docker run -d --gpus all -p 8765:8888 l2g