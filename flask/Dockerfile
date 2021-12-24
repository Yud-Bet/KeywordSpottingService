FROM tensorflow/tensorflow:2.6.0

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["uwsgi", "app.ini"]