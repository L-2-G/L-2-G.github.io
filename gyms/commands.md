# Create the Docker Image:

cd l-2-g.github.io


docker build gyms --tag l2g

# Run the Docker Image:

docker run -d --gpus all -p 8765:8888 l2g


The -p argument is to connect the docker vm port to your local port so your notebook would be at localhost:8765

If you are using a remote machine you may have to do additional port-forwarding