FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install development tools including gcc (includes the GNU C++ compiler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

# Install swig
RUN apt-get update && apt-get install -y swig
RUN apt-get install -y tree

# Install dependencies for rendering graphics
RUN apt-get install -y xvfb x11-utils

# Copy the entire project code to the container
COPY . .
RUN rm -rf /app/.venv   # Except this

# Set PYTHONPATH to include the directory containing the 'final_project' module
ENV PYTHONPATH="${PYTHONPATH}:/app/bachelorproject"

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

RUN poetry lock
# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi

CMD ["python", "bachelorproject/experiment.py"]