FROM python:3.10.12-slim
WORKDIR /app
RUN mkdir -p /app/models_binary

# Set the working directory to /app
WORKDIR /app

# Copy the model, DictVectorizer, and StandardScaler files into the models_binary directory

COPY requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server.py file into the container
COPY model ./model
COPY generate.py ./

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "generate:app", "--host", "0.0.0.0", "--port", "8000"]
