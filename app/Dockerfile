FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm

COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:create_app()"]
