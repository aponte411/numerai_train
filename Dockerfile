FROM python:3.7

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

CMD [ "python", "predict.py", "--model catboost", "--load-model False --save-model True"]
