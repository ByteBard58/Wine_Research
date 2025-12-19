# Use an official Python runtime as a parent image
# python:3.10-slim is a good balance of size and compatibility and supports multi-arch
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application using Gunicorn
# Binding to 0.0.0.0:5000 to be consistent with the exposed port
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
