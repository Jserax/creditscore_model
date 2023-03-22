FROM python:3.9-slim

COPY requirements.txt app/requirements.txt 

RUN pip3 install -r app/requirements.txt

COPY run_model.py app/run_model.py
COPY models/model.joblib app/models/model.joblib
