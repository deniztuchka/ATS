FROM python:3.11-slim
WORKDIR /project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /project/app
COPY . /project
ENV PORT=8000
EXPOSE 8000
CMD ["gunicorn", "app.main:create_app()", "--bind", "0.0.0.0:8000", "--chdir", "/project"]
