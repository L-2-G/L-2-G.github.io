FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY . .

RUN conda install cudatoolkit

RUN conda install numba

RUN pip3 install stable-baselines3

RUN pip3 install jupyter

RUN pip3 install ipycanvas

CMD [ "python3","-m","notebook","--allow-root","--ip","0.0.0.0","--no-browser"]