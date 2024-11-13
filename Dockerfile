# Base image with Python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the application code and files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the necessary ports for both Flask and Streamlit
EXPOSE 5000 8501

# Environment variable to set which app to run (Flask by default)
ENV APP_MODE=flask

# Command to run either the Flask or Streamlit app based on APP_MODE
CMD if [ "$APP_MODE" = "flask" ]; then \
    python src/flask_app.py; \
    else \
    streamlit run src/streamlit_app.py --server.port 8501 --server.address 0.0.0.0; \
    fi