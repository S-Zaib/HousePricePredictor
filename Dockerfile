# Use the official Python 3.11 image from Docker Hub as the base image
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the Python dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the port that Flask will be running on (5000)
EXPOSE 5000

# Set environment variable for Flask default to production
ENV FLASK_ENV=production

# Use Gunicorn for production and Flask's built-in server for development
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
