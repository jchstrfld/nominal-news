# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y git

# Copy files into the container
COPY . .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Run the app
ENTRYPOINT ["python"]
CMD ["generate_daily_news_email.py"]


