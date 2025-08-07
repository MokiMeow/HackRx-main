# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
# This is important for libraries like opencv-python and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY ./src ./src

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run your app using uvicorn+
# We use --host 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
