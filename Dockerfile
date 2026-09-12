# 1. Use an official Python runtime as a parent image
FROM python:3.12-slim
# 2. Tell Python not to buffer output
ENV PYTHONUNBUFFERED=1
# 3. Set the working directory in the container
WORKDIR /app
# 4. Copy and install requirements
COPY steer3dprj/requirements.txt .
# Install dependencies if present
RUN if [ -f requirements.txt ]; then \
      pip install --no-cache-dir -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
      pip install --no-cache-dir .; \
    else \
      echo "No requirements.txt or pyproject.toml found, skipping deps install."; \
    fi
# 5. Copy the rest of the application code
COPY . .
# 6. Run main.py when the container launches
CMD ["python", "main.py"]

""" Run Senior way
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \...
"""
