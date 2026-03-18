FROM python:3.11-slim

WORKDIR /app

COPY scal_webapp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scal_webapp /app/scal_webapp

EXPOSE 8080
CMD ["uvicorn", "scal_webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
