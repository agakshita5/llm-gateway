FROM python:3.13-slim

# no .pyc files, and logs go straight to docker instead of sitting in a buffer
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

# install dependencies first, in their own layer: code edits then don't re-run pip
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py && pip install --no-cache-dir . && rm -rf app

COPY app ./app
RUN pip install --no-cache-dir --no-deps .

# don't run as root inside the container
RUN useradd --create-home gateway
USER gateway

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
