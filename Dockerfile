FROM python:3.10-slim

WORKDIR /app

COPY /requirements.txt ./requirements.txt
COPY /ML_Algorithm ./ML_Algorithm
COPY /database ./database
COPY /new_duplicate ./new_duplicate
COPY /app.py ./app.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]