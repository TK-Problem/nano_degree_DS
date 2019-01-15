# Udacity Data Scientist Nano Degree

Personal repo for projects Udacity data scientist nano-degree.

PATH to venv python interpreter `"...\venv\Scripts\python.exe"`

# Project structure

Files are organized in following folders.

## 1. Supervised Learning

### Lesson 2

**Linear Regression**.

* `Quiz_16` manual implementation of mini-batch gradient descent,
* `Quiz_18` linear regression implementation using sklearn package,
* `Quiz_20` multiple Linear Regression implementation using sklearn package,
* `Quiz_25` polynomial Regression using sklearn package,
* `Quiz_27` L1 regularization using sklearn package,
* `Quiz_28` Standard scaler implementation using sklearn package.

### Lesson 3

**Perception algorithm**.

* `Quiz_7` simple quiz, which asks to set weights and bias to correctly determine the AND operation,
* `Quiz_9` manual implementation of perception algorithm.

### Lesson 4

**Decision tree**

* `Quiz_16` manually calculated information gain for given `ml-bugs.csv` dataset,
* `Quiz_18` decision tree implementation using sklearn,
* `Quiz_19` titanic survival data study using sklearn decision trees model.

### Lesson 5

**Naive Bayes**

* `Quiz_15` Naive Bayes implementation using scikit-learn for spam classifier. 

### Lesson 6

**Support vector machines**

* `Quiz_17` Support vector machines implementation in sklearn.

### Lesson 7

**Ensemble methods**

* `Quiz_12` Bagging, RandomForest, AdaBoost classifiers implementation using scikit-learn for spam classifier. 

### Lesson 8

**Model evaluation metrics**

* `Quiz_3` calculate the accuracy using sklearn.metrics,
* `Quiz_16` accuracy score evaluation for spam classifier using Bagging, RandomForest, AdaBoost classifiers,
* `Quiz_18` manual implementations for r2 score, mean square error and mean absolute error.

### Lesson 9

**Training and tuning models**

* `Quiz_10` example of improving model using Grid search,
* `Quiz_12` diabetes case study using various sklearn classifiers and grid search.

### Project #1

**Finding Donors for CharityML**

* `finding_donors.ipynb` is final submitted notebook
* `cencus.csv` the project dataset,
* `visuals.py` additional Python script provided by Udacity which adds supplementary visualizations for the project.


## 2. Deep Learning

### Lesson 1

**Introduction to Neural Networks**

* `Quiz_21` manual implementation of cross-entropy equation,
* `Quiz_27` manual Gradient Descent Algorithm implementation,
* `Quiz_36` predicting student admissions using manually implemented neural networks.

### Lesson 2

**Implementing Gradient Decent**

* `Quiz_4` manual gradient decent implementation using sigmoid and sigmoid prime functions,
* `Quiz_5` implementing gradient decent using numpy,
* `Quiz_6` manual neural network multilayer implementation,
* `Quiz_7` backpropagation calculation,
* `Quiz_8` manual backpropagation implementation.

### Lesson 4

**Keras package**

* `Quiz_2` create basic Keras multilayer model,
* `Quiz_8` classify IMDB review data using Keras.

### Lesson 5

**PyTorch package**

Folder contains iPython notebooks, where PyTorch package is used to classify images (MNIST fashion and handwritten numbers training dataset).

### Project #2

Image classification project- build and train PyTorch convolution network to classify 106 types of flowers.

## 3. Unsupervised Learning

### Lesson 1

**Clustering**

K-means implementation using sklearn package. Examples, how results are affected by changing k-means parameters. Also, the importance of feature scaling is shown. 

### Lesson 2

**Hierarchical and density based clustering**

* `Quiz_7` Hierarchical clustering implemented on [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris),
* `Quiz_13` DBSCAN (density based spatial-clustering applications with noise) implementation on various datasets.

### Lesson 3

**Gaussian mixture models and clustering validation**

* `Quiz_21` GMM (Gaussian mixture models) vs K-means on generated dataset. Visual illustration how results are affected by different unsupervised learning approaches.

### Lesson 4

**Dimensionality reduction**

Various examples of PCA (principle component analysis) are provided.

### Lesson 5

**Random projection**

`Independent Component Analysis Lab.ipynb` shows how ICA (independent component analysis) can be used to filter out 3 instruments from noisy dataset.

### Project #3

Unsupervised learning techniques are applied on Bertelsmann partners AZ Direct and Arvato Finance Solution customer dataset. In this project project population was grouped into various customer segments.

## 4. Data Science process

### Lesson 1

**CRISP-DM** (Cross Industry Process for Data Mining). Folder contains Jupyter notebooks about data exploration, preparation, modeling and evaluating results. In all notebooks stack overflow survey data was analyzed. 

In `Putting It All Together.ipynb` notebook, Linear Regression model is used to identify most important salary 
predicting features. 

### Project #4

[Blog post](https://medium.com/@t.uzdavinys/statistical-analysis-of-nba-odds-how-to-not-lose-money-betting-on-basketball-bc41fe239561) about NBA odds was written. Data processing code is available on separate [Github](https://github.com/TK-Problem/Interesting_Sport_Stats) repo.

## 5. Software Engineering

### Lesson 2

In this lesson, example code was refactored to increase performance and readability.

## 6. Data Engineering

### Lesson 2

**ETL** (extract transform load) pipeline. Folder contains Jupyter notebooks about:
 
* loading data from various file formats,
* transforming data (cleaning missing values, creating dummy variables, scaling features, ect.),
* loading data into SQLite database.

### Lesson 3

**NLP** (natural language processing) pipeline. Folder contains Jupyter notebooks about:

* webscraping and processing text of Udacity course website,
* normalize text (lower case words, remove punctuation, ect.),
* tokenization (split text into words or sentences),
* Part-of-speech tagging and named entity recognition,
* stemming and lemmatization.

### Lesson 4

Machine Learning (ML) pipeline. In `grid_search.ipynb` notebook, ML pipeline with Feature union and Grid search was built to classify twitter text messages. Other notebooks contain examples of different parts of ML pipeline.

### Project #5

# 7. Experimental Design & Recommendations

### Lesson 3

**Statistical considerations in testing**

Folder contains notebooks with examples how to calculate *p*-values for different tests, how to decide on experiment size and when to stop it if needed.

### Lesson 6

**Introduction to recommendation engines**

Notebooks analyze movie review data and provide recommendations for users based on Similarity (Pearson's correlation coefficient and Euclidean distance) or Neighborhood Based Collaborative Filtering (recommends similar movies user already watched) .

## Documents

`Data+Scientist+Nanodegree+Syllabus.pdf` contains syllabus for Data Scientist nanodegree and `ud123-git-keyterms.pdf`
contains description of key terms for Git version control.

# Requirements

Requirements are available at `requirements.txt` file.