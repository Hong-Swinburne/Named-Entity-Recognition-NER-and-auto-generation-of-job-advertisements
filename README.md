# Named Entity Recognition (NER) and auto-generation of job advertisements

This repo implemented two major NLP tasks, e.g. NER and automatic generation of advertisement, for job advertisements using Python libraries.The dataset used in the case study, ads-50k.json, has 50,000 job ads stored in json format, in which the ad information is encoded in a dictionary containing 5 fields.
* `"id"`: ad ID
* `"title"`: ad title
* `"abstract"`: abstract of job description
* `"content"`: detailed job description
* `"meta"`: meta information (e.g. location, classification, work type ) of ad
Because the **content** field is the most important field that describes the detailed information of the job ad. So we only focus on this filed in data pre-processing and exploration, as well as the task of auto recognition of the named entities.

In the NER task, customised named entities, such as "*responsibilities*", "*skills*", "*experience*", "*requirements*", and "*qualifications*" will be automatically annotated in job advertisements using the trained NER model. In the auto-generation of job ads task, the generation model will automatically generate high-quality job advertisements in terms of conciseness and written styles

## Prerequisites
* Python (3.7)
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Ipython
* Jupyter
* Nltk
* Wordcloud
* Torch
* Transformers
* SpaCy (3.1)
  
Run `pip install -r requirements.txt` to install the required packages

## Dataset
The data used in this project can be accessed via [google drive](https://drive.google.com/drive/folders/1VM_PWssURRrwGhuaWMPxb88_aOWsLG7P?usp=sharing)

## Data pre-processing and exploration
In data pre-processing and exploration, we mainly perform low-level text pre-processing operations on job ads. Our pipelines comprise the following steps:
* **Text cleaning**: clear HTML tags, remove special characters and formatting symbols
* **Tokenize**: sentence tokenization, word tokenization, stop word removal
* **Lemmatization**
* **POS Tagging**: identify POS of interest, POS category counting
* **Word/sentence counting**: count words/sentences in each ad
* **Frequent Words**: BiGrams, TriGrams ranking

Refer to "Preprocessing_EDA.ipynb" for more details.

## Automatic annotation of named entities
Five types of named entities, e.g. "**responsibilities**", "**skills**", "**experience**", "**requirements**", and "**qualifications**", are annotated in the job advertisements. We trained the NER model on a selected subset of job ads that contain "*machine learning*", or "*data science/scientist*", or "*AI*" in the job contents. [UBIAI](https://ubiai.tools/) application was used to annotate the job ads. Refer to "NER_model.ipynb" for more information.

## Auto-generation of job advertisements
The purpose of this task is to assist advertisers to automatically generate better written and concise job ads. For this purpose, the qualities of job ads were annotated in terms of conciseness and written style. GPT-2 model implemented by [Huggingface](https://huggingface.co/gpt2?text=A+long+time+ago%2C) was applied on the job advertisement dataset to fine-tune the ads generation model. Refer to "ads_generation.ipynb" for more information.

## Deployment of NER model using Flask and Docker
1. Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop) on your machine
2. Create a folder to contain the required files for model deployment. In this repo, the folder `"\Docker"` includes the following necessary files/folders for model deployment
      * **`\output`**: Trained NER model files
      * **`\templates`**: HTML templates used for REST APIs in Flask
      * **Dockerfile**: Dockerfile
      * **ner_app.py** Inference code of NER model
      * **requirements.txt**: Required python packages for model inference
      * **ad_xxxx.txt**: Example files of job advertisements for model inference

3. Change path to `"\Docker"` folder and build Docker image on local machine by running
    > cd Docker

    > docker build -t ner_app_docker_image .

    After executing the above command in terminal, you will have a Docker image tagged with `ner_app_docker_image`. You may list all available Docker images on your machine using `docker image ls`

4. Run Docker container on local machine
    > docker container run -p 5000:5000 ner_app_docker_image

5. Open browser and input `localhost:5000` to test NER model on local machine. You may view the running Docker container by executing `docker ps` in terminal
6. Stop the running Docker container
    > docker stop containerID

    containerID is the CONTAINER ID of **ner_app_docker_image** which you may obtain by running `docker ps`
7. Push Docker image to Docker Hub (docker.io)
   * Login Docker Hub using your userID and password in CLI
        > docker login -u "userID" -p "password" docker.io

   * Build a Docker image and tag it with your username and image name (ner_app_docker_image)
        > docker build -t username/ner_app_docker_image .

   * Push Docker image to Docker Hub
        > docker push username/ner_app_docker_image