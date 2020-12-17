FROM python:3.8

WORKDIR /app

RUN mkdir ./mlsample

RUN mkdir ./mlsample/model

ENV MODEL_DIR=/app/mlsample/model

ENV MODEL_FILE=/model.joblib

ENV FLASK_APP=app.py

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN python model.py

EXPOSE 5000

EXPOSE 8000

CMD ["python", "app.py"]
