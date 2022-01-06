FROM python:3.8

RUN ["pip", "install", "tensorboard"]

CMD [ "tensorboard", "--logdir", "runs", "--host", "0.0.0.0", "--port", "5001"]