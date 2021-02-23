## Introduction
In this project, I applied Data Engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies 
disaster messages.A data set containing real messages that were sent during disaster events, has been used to create a machine learning pipeline. 
This pipeline is used to categorize these events so that the messages can be directed to an appropriate disaster relief agency.
A web app has also been included where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data. 

## Project Components
### ETL Pipeline

The Python script, process_data.py contains data cleaning pipeline that
* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

### ML Pipeline

The Python script, train_classifier.py, contains a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

### Flask Web App
Flask, html, css and javascript knowledge have been utilized to create the web app.
Plotly has been in a great use of building data visualizations in the web app. 

### data
This folder contains sample messages and categories datasets in csv format.
### app: 
This folder contains all of the files necessary to run and render the web app.

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
* Go to http://0.0.0.0:3001/ (If this one doesn't work, follow the below instructions)
* In the terminal, use this command to get the link for vieweing the app:
env | grep WORK
The link wil be:
http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.


## Screenshots
### Screenshot_1 

<img width="898" alt="Screenshot_1" src="https://user-images.githubusercontent.com/39970140/108806394-6d84f780-7556-11eb-8ef8-852336776d5e.png">

### Screenshot_2

<img width="891" alt="Screenshot_2" src="https://user-images.githubusercontent.com/39970140/108806540-d10f2500-7556-11eb-9af6-dadd85d2220b.png">


### Screenshot_3

<img width="595" alt="precision_recall_Fscore_report" src="https://user-images.githubusercontent.com/39970140/108806689-411dab00-7557-11eb-9fa5-5472005fbaca.png">


### Screenshot_4

<img width="351" alt="Screenshot_4" src="https://user-images.githubusercontent.com/39970140/108806738-601c3d00-7557-11eb-9b32-b986c9097ffa.png">

### Screenshot_5

<img width="393" alt="Screenshot_5" src="https://user-images.githubusercontent.com/39970140/108806776-7f1acf00-7557-11eb-8659-46c5ca79acfe.png">


## Analysis of results and model
As we can see from the visualization that the dataset used for the analysis and model building is really imbalanced. There are categories those are very few in numbers(less positive) can lead to incorrect insight gathering and prediction.In such cases, even though the classifier accuracy is very high (since it tends to predict that the message does not fall into these categories), the classifier recall (i.e. the proportion of positive examples that were correctly labelled) tends to be very low. While using this project,one should take this into consideration that applying measures like synthetic data generation, model selection and parameters fine-tuning, etc can help in predicting better.

## Acknowledgement
I am really grateful to be a part of this Udacity DataScince Nanodegree Program. This lesson of Data Engineering Skills provides a great bunch of opportunities to apply them in real life problem solving such as this disaster response pipeline. 
Thanks goes to Figure Eight for the dataset. This was explored to build a model for an API that classifies disaster messages.

## References
Inspiration for building the pipeline and the model to create the webapp is supported by the following q/a and github project.
1. https://knowledge.udacity.com/questions/306891
2. https://knowledge.udacity.com/questions/140072
3. https://knowledge.udacity.com/questions/327812
4. https://github.com/gkhayes/disaster_response_app

## License
License: MIT
