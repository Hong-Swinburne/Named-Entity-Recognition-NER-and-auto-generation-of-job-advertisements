FROM python:3.6
COPY . /usr/NLP_NER/app
EXPOSE 5000
WORKDIR /usr/NLP_NER/app
RUN pip install -r requirements.txt
CMD ["python", "ner_app.py"]