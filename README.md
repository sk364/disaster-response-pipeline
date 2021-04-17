## Table of Contents

* [Installation](#installation)
* [How To Run](#run)
* [Project Motivation](#motivation)
* [File Descriptions](#desc)
* [Results](#results)

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

## Results<a name="results"></a>

...
