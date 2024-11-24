# Use Python base image
RUN pip install uvicorn
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
