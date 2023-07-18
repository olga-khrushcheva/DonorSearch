FROM python:3.9.6
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD python web_app.py