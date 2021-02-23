## Introduction
In this project, I applied Data Engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies 
disaster messages.A data set containing real messages that were sent during disaster events, has been used to create a machine learning pipeline. 
This pipeline is used to categorize these events so that the messages can be directed to an appropriate disaster relief agency.
A web app has also been included where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data. 

## Project Components

1. ETL Pipeline

The Python script, process_data.py contains data cleaning pipeline that
* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. ML Pipeline

The Python script, train_classifier.py, contains a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Flask Web App

Flask, html, css and javascript knowledge have been utilized to create the web app.
Plotly has been in a great use of building data visualizations in the web app. 

## Software needed
The code contained in this repository was written in HTML and Python 3, and requires the following Python packages: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle, warnings.

## Running Instructions
1. Run process_data.py
* Save the data folder in the current working directory and process_data.py in the data folder.
* From the current working directory, run the following command: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

2. Run train_classifier.py
* In the current working directory, create a folder called 'models' and save train_classifier.py in this.
* From the current working directory, run the following command: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

3. Run the web app
* Save the app folder in the current working directory.
* Run the following command in the app directory: python run.py
* Go to http://0.0.0.0:3001/

## Screenshots
Screenshot_1

Screenshot_2

Screenshot_3

Screenshot_4

Screenshot_5

## Acknowledgement
I am really grateful to be a part of this Udacity DataScince Nanodegree Program. This lesson of Data Engineering Skills provides a great bunch of opportunities to apply them in real life problem solving such as this disaster response pipeline. 
Thanks goes to Figure Eight for the dataset. This was explored to build a model for an API that classifies disaster messages.

## License
License: MIT
