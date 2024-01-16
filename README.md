# Song Lyric Generation Project

## Problem Description

In this project, which is a part of my journey in [ML zoom camp](https://github.com/DataTalksClub/machine-learning-zoomcamp), I tackle the  challenge: "Can I create a machine learning model that generates song lyrics in the style of a specific artist?" My goal is to replicate the lyrical style of Taylor Swift, known for her unique narrative and emotive songwriting.

## Word-Level Modeling

The project focuses on word-level modeling to generate lyrics. This approach involves training a neural network model to understand and produce lyrics by learning from a dataset of existing songs. Word-level modeling is particularly interesting in this context because it allows the generated lyrics to maintain coherent structure and meaning, closely resembling the style of the chosen artist.

While this project focuses on word-level modeling, it's noteworthy that another prevalent approach in text generation tasks is character-level modeling. Character-level models generate text one character at a time and can capture unique stylistic and structural elements of writing, such as the use of specific punctuation and capitalization patterns. For those interested in exploring this alternative method, an example of character-level lyric generation can be found in this [Kaggle notebook](https://www.kaggle.com/code/karnikakapoor/lyrics-generator-rnn).

## My Plan of Action

The workflow of the project is structured as follows:

1. **Data Collection and Preprocessing**: Assembling a comprehensive dataset of Taylor Swift's songs and preprocessing the text to make it suitable for machine learning.

2. **Model Building and Training**: Using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network, to capture the sequential nature of lyrics. The model is trained on the processed dataset, learning the patterns and styles of the artist's writing.

3. **Lyric Generation and Evaluation**: After training, the model is used to generate new lyrics. The success of the model is evaluated based on the creativity, relevance, and stylistic similarity of the generated lyrics to Taylor Swift's original songs. It's important to note that evaluating the success of such a creative endeavor is inherently subjective and can best be assessed by human judgment, as current automated evaluation metrics may not fully capture the nuances of artistic expression in songwriting.

4. **Iteration and Improvement**: Continuously refining the model by experimenting with different architectures, hyperparameters, and training techniques to enhance the quality of the generated lyrics.

## Dataset

The dataset used for this project is a comprehensive collection of Taylor Swift's song lyrics, spanning all her albums. This dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/taylor-swift-song-lyrics-from-all-the-albums/data), and it provides a rich source of text data for training our lyric generation model.

You can find this dataset in `data/taylor_swift_lyrics.csv`

It includes the following data fields:

- Album Name: The name of the album to which the track belongs.
- Track Title: The title of the track.
- Track Number: The sequence number of the track in the album.
- Lyric Text: The actual lyrics of the song.
- Line Number: The line number of the lyric in the track.
- Year of Release: The year when the album was released.

I found out that this dataset contains only her first 6 albums so I used Genius API to scrape additional 4 albums. You can find a scrapper and explanation in `notebooks/scrapper.ipynb` and a separate dataset `data/taylor_swift_4_albums.parquet`

## Data Preparation

In order to use this dataset for training a word-level LSTM model, significant data preparation is required:

1. **Data Cleaning**: The raw lyrics data is cleaned to remove any extraneous characters or symbols that might interfere with the learning process of the model.

2. **Tokenization**: The lyrics are tokenized into words. This process involves splitting the text into individual words and mapping them to integers, creating a vocabulary of unique words.

3. **Sequence Generation**: The tokenized lyrics are then used to generate sequences of words. Each sequence serves as input to the model, with the model trained to predict the next word in the sequence.

4. **Data Splitting**: The prepared dataset is split into training and validation sets. The training set is used to train the model, while the validation set helps in evaluating the model's performance and tuning its hyperparameters.

## Instructions on How to Run the Project

### Setting up the Environment

1. First, ensure you have [pyenv](https://github.com/pyenv/pyenv) installed on your system.

2. Install Python 3.11.4 using pyenv:

    ```bash
    pyenv install 3.11.4
    ```

3. Create a new virtual environment for this project:

    ```bash
    pyenv virtualenv 3.11.4 lyrics-generation
    ```

4. Set the local Python version to use the virtual environment you just created:

    ```bash
    pyenv local lyrics-generation
    ```

5. Now, the project directory is set up with a local virtual environment using Python 3.11.4.

### Installing Dependencies

1. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Setting up Visual Studio Code (VSCode)

1. Open the project directory in VSCode.
2. Ensure you have the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) installed in VSCode.
3. You can now create and open Jupyter Notebooks within VSCode.
4. When you run Jupyter Notebook in VSCode choose lyrics-generation Python environment from drop-down list of avaliavle environments.

### Setting up Jupyter Notebook Independently

1. If you prefer to use Jupyter Notebook outside of VSCode, ensure you have Jupyter installed:

    ```bash
    pip install jupyter
    ```

2. Launch Jupyter Notebook from the project directory:

    ```bash
    jupyter notebook
    ```

3. Jupyter Notebook will open in your web browser, and you can create new notebooks or open existing notebooks from the browser interface.

### Setting up Google Colab

I run the model notebook using Google Colab. To replicate this, you'll need to upload the `model.ipynb` notebook to Google Colab and also upload the `cleaned.parquet` file to the 'Files' section in your Colab environment.

<img width="507" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/45e7ac95-9673-429d-8c26-647a140d5477">

## Exploratory Data Analysis (EDA)

The exploratory data analysis is detailed in **notebooks/EDA.ipynb**, using the dataset located at `data/taylor_swift_lyrics.csv`.

In the end, the cleaned dataset was saved to **data/cleaned.parquet** for use in Modeling.

## Model Selection Process and Parameter Tuning

In **notebooks/model.ipynb**, I explore a variety of models suitable for lyric generation task.

### Model Selection

The process of selecting the right model for this lyric generation task involved experimenting with various types of neural network architectures. Given the sequential nature of text data, the focus was primarily on Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models, known for their effectiveness in capturing long-range dependencies in sequences.

- **LSTM**: Initially, I experimented with LSTM models due to their ability to retain information over longer sequences, making them well-suited for text generation tasks.
- **GRU**: I also explored GRU models, which are similar to LSTMs but with a simpler structure. They often provide comparable performance with reduced computational complexity.
- **Bidirectional Models**: Additionally, bidirectional versions of these RNNs were tested, as they process data in both forward and backward directions, potentially providing a more comprehensive understanding of the context.

## Long Short-Term Memory (LSTM) Models

Long Short-Term Memory (LSTM) models are a type of Recurrent Neural Network (RNN) specialized in processing sequences of data. They are particularly adept at handling long-range dependencies in sequential data, which makes them well-suited for applications in natural language processing, time series analysis, and more.

### Key Features of LSTM

- **Memory Cells**: At the heart of LSTM models are memory cells that can maintain information in memory for long periods. This is crucial for understanding context in text or patterns in time series.

- **Gates**: LSTMs have a unique structure consisting of gates (input, forget, and output gates) that regulate the flow of information into and out of the memory cell. This gating mechanism helps the model decide which information to keep, discard, or pass on to the next time step.

- **Handling Vanishing Gradient Problem**: Traditional RNNs often struggle with the vanishing gradient problem, where they lose track of long-range dependencies in a sequence. LSTMs overcome this by their ability to retain information over long sequences, making them more effective for complex tasks.

### Parameter Tuning

The performance of these models highly depends on their configuration. Therefore, parameter tuning was a crucial part of the model development process:

- **Number of Layers and Units**: Different configurations of the number of layers and the number of units per layer were tested to find the optimal balance between model complexity and performance.
- **Embedding Layer**: For word-level models, I experimented with different sizes of embedding layers, including using pre-trained word embeddings like GloVe, to capture more nuanced word relationships.
- **Dropout and Regularization**: To prevent overfitting, I used dropout layers and experimented with their rates.
- **Learning Rate and Optimizer**: Different learning rates and optimizers were trialed to identify the most effective combination for training the models.

### Evaluation and Iteration

The evaluation of the models extended beyond traditional metrics to include a more nuanced assessment of the generated lyrics:

- **Quantitative Metrics**:

  - **Loss and Accuracy**: These standard metrics were used to gauge the model's learning efficiency and its ability to accurately predict the next word in a sequence.
  - **Perplexity**: This is a crucial metric in language modeling, measuring how well a probability model predicts a sample. Perplexity essentially quantifies the uncertainty of the language model in predicting the next word. A lower Perplexity score indicates that the model is more confident in its predictions, which typically correlates with better text generation.

- **Temperature-Based Text Generation**: During the text generation phase, the concept of 'temperature' was employed to control the randomness in the prediction process. A higher temperature results in more varied and creative outputs, whereas a lower temperature yields more conservative and expected text.

## Training the Final Model

I copied the code from data cleaning and features extraction process into `train.py` file.

It loads the dataset from  `data/cleaned.parquet`, cleans it and extract text. Then it train the model.

How to run:

```python
python train.py
```

## Loading the Model and Serving It via a Web Service

Finally, I've set up a web service (Fast API) to serve the trained model.

The code for the service is located in the `generate.py` file.

### How to Run the Service

To run the web service, use the following command in the terminal:

```bash
uvicorn generate:app --reload
```

Then open `generate_request.ipynb`. Here you can find code for request to the server (sort of a client) and JSON sample:

```python
data = {"prompt": "ice on a hand", "temp": 0.3}
```

Here prompt field is a seed phrase for your new lyrics and temp field is temperature between 0 and 1. A higher temperature results in more varied and creative outputs, whereas a lower temperature yields more conservative and expected text.

Click run

<img width="949" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/26b197b5-983c-445a-9aa9-25065a31f203">

You should see 200 result on a server side

<img width="1184" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/4e23d968-a654-4757-8a8e-4c0d09ce320d">

And then you get the result of generation in Jupyter notebook. Example:

```json
{"result": "ice on a hand and i am a getaway car fake fake fake fake fake fake fake hardwood cages tendency playboy counter fresh hits state sleepless pointed bunch wednesday burnin' liars gary chandelier punched sixties' vanished and you are the la la marvelous la marvelous mm la la marvelous motown la la marvelous la marvelous letter familiar rides gown dope apologies weekend organ  roller sights boyfriend \x91cause playful hehe ladadada counting feast shot and bleachers staring playboy familiar will x streak to me up tragic hindsight stay and you are the one time are the way you are is a little ground bridesmaid"}
```

## Docker

To containerize and run the model locally using Docker, you'll need to follow these steps:

### 1. **Install Docker Desktop for Mac**

- Go to the [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop) download page.
- Click on "Download for Mac" and the download will start automatically.
- Once downloaded, double click on the Docker.dmg file to open it.
- Drag the Docker icon to your Applications folder to complete the installation.
- Launch Docker Desktop from your Applications folder.

### 2. **Build the Docker Image**

- Open a terminal.
- Navigate to the project root directory midterm_project.
- Run the following command to build `lyric-generator` Docker image:

```bash
docker build -t lyric-generator .
```

### 3. **Run the Docker Image**

- After successfully building the image, run the following command to start a container from the image

```bash
docker run -p 8000:8000 lyric-generator
```

Wait a bit because it takes time for tensorflow to start.

### 4. **Accessing the Service**

- Now that the model is running in a Docker container, open `midterm_project/generate_request.ipynb`. Here you can find code for request to the server.

<img width="561" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/978d1a5d-892b-4d46-bb1e-8263612f7a30">

<img width="1181" alt="image" src="https://github.com/olgazju/ml_camp_2023/assets/14594349/9dbbf446-a8d1-4991-82bd-a4ef7f0cc552">

### 5. **Stop the Docker Container**

- To stop the Docker container, find the container ID with the following command:

```bash
docker ps
```

Then stop the container with:

```bash
docker stop <container-id>
```

## Cloud Deployment

### Installing and Running Minikube

Minikube is a tool that lets you run Kubernetes locally. Minikube runs a single-node Kubernetes cluster on your personal computer (including Windows, macOS, and Linux PCs) so that you can try out Kubernetes, or for daily development work.

1. **Install Homebrew** (if it's not already installed):

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. **Install Minikube via Homebrew**:

    ```bash
    brew install minikube
    ```

3. **Start Minikube**:

    ```bash
    minikube start
    ```

4. **Check Minikube Installation**:

    ```bash
    minikube status
    ```

5. **Install kubectl**:

   ```bash
       brew install kubectl
    ```

6. **To point your terminal to use the docker daemon inside minikube run this**

   ```bash
       eval $(minikube docker-env)
    ```

   Now any ‘docker’ command you run in this current terminal will run against the docker inside minikube cluster.

7. **To build docker image inside minikube**

   ```bash
       minikube image load python:3.10.12-slim
       docker build -t lyric-generator . --progress=plain
    ```

   If I run

   ```bash
       docker ps
    ```

   I should see my lyric-generator:latest image.

8. **Let's deploy the model**

   You have deployment.yaml file in the project folder. Deploy it and check if it went well.

   ```bash
       kubectl apply -f deployment.yaml
       kubectl get deployments
    ```

9. **Expose the application**

   Create a Kubernetes service:

   ```bash
       kubectl expose deployment fastapi-lyric-generator-deployment --type=NodePort --port=8000
    ```

   Find the Minikube Service URL. The command below will open a browser with the right link for your service. Copy this link, open `generate_request.ipynb`. Here you can find code for lyrics generation and insert your url in the code same way as it was descibed for fastapi service and docker container.

   ```bash
       minikube service fastapi-lyric-generator-deployment
    ```

10. **To stop minukube**

   ```bash
       minikube stop
   ```

#### What is this deployment.yaml file?

This YAML file is for a Kubernetes deployment configuration named `fastapi-lyric-generator-deployment`. It's designed to deploy a FastAPI application containerized with the Docker image `lyric-generator:latest`. The deployment ensures that one replica of the application is running (`replicas: 1`). It uses the label `app: fastapi` to manage and identify the pods. The `imagePullPolicy: IfNotPresent` setting is used to avoid pulling the image from a remote registry if it already exists locally. The application listens on port 8000 inside the container, aligning with the FastAPI server's default port.
