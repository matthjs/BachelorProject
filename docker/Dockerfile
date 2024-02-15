FROM pytorch/pytorch:latest as python-base

# Set the working directory in the container
WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

RUN poetry lock
# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi
# Copy the entire project code to the container
COPY . .

CMD ["python", "other/pytorch-gpu-test.py"]
