# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and model file into the container
COPY app.py .
COPY ./model/brain_tumor_model.h5 ./model/brain_tumor_model.h5

# Expose port 5000
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]