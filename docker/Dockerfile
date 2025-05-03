# using python slim image
FROM python:3.8-slim

# Install required packages
RUN pip install --no-cache-dir tensorflow

# Set the working directory
WORKDIR /app

# Copy the training script into the container
COPY train.py .

# Define the default command to run the training script
ENTRYPOINT ["python", "train.py"]
