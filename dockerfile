FROM python:3.9.12-slim

WORKDIR /app

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install gdown

RUN  gdown --fuzzy https://drive.google.com/file/d/1EBYQN9evN9lXOLaGqah29k7mOxlvcrrW/view?usp=sharing

COPY . .

EXPOSE 5000

ENTRYPOINT [ "python3", "app.py" ]