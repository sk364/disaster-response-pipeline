## Table of Contents

* [Installation](#installation)
* [How To Run](#run)
* [Project Motivation](#motivation)
* [File Descriptions](#desc)
* [Screens](#scr)
* [Acknowledgements](#ack)

## Installation<a name="installation"></a>

Install the packages in a Python 3.7+ virtual env using the following command:
```
pip install -r requirements.txt
```

## How to run?<a name="run"></a>

```
cd data
python process_data.py messages.csv categories.csv DisasterResponse.db MessageCategories

cd ../models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl

cd ../app
python run.py
```

Go to `http://localhost:3001`.


## Project Motivation<a name="motivation"></a>

The project is developed for the interest to create an efficient pipeline to categorize real-world messages for a disaster, routing the message to an appropriate disaster relief agency, predicting the category of the message using a text processing machine learning model.

## File Descriptions<a name="desc"></a>

The repository holds 3 directories:

* app
* data
* models

`app` directory contains the code to run the Flask app.
`data` contains the CSV datasets, data processing script and processed SQLite database file.
`models` contains the classifier training script and the resulting model from it.

## Screens<a name="scr"></a>

**Genre and Category Distribution**:
![Landing Page](./assets/main-page.png)

**Classification of a message**:
![Classification](./assets/classify-1.png)
![Classification](./assets/classify-2.png)

## Acknowledgements<a name="ack"></a>

Thanks to Figure Eight for providing the disaster messages and categories datasets and a huge shoutout to Udacity for the course content, guiding me well to develop this project.
