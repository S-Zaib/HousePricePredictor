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

# Set environment variable to ensure Flask runs in production
ENV FLASK_ENV=production

# Command to run your Flask app
CMD ["python", "app.py"]

