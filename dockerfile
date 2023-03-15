FROM python:3.9-slim

COPY requirements.txt app/requirements.txt 

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY run_model.py app/run_model.py
COPY models/model.joblib app/models/model.joblib

EXPOSE 5000

CMD ['python3', 'run_model.py']