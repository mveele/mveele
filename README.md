# About Me

Hi, I'm [Mitch Veele](https://www.linkedin.com/in/mitch-veele/) - I'm a master's student in Data Science at the University of San Francisco, graduating in July 2022. I'm interested in Machine Learning Engineer, Data Scientist, and Data Engineer roles. My consistent trait throughout my work experience has always been to improve processes - such as organizing the tools and maintenance schedule in the electrician's shop or scraping web pages to automate a data process. Data Science is just the next step forward - building better data pipelines and automating the decision-making process. Please take a look at some of the projects below - I'm happy to send over anything that sparks your interest.

# Projects
Note: all projects from my school below are private - please get in touch with me for samples of my work.  

## Machine Learning

##### **K-means Clustering**
Implemented the k-means clustering algorithm with the k-means++ initial centroid method in NumPy. Showcased k-means strengths with color image compression. Demonstrated algorithmic speed improvements by vectorizing all functions - with 100x speed improvement for image compression.

##### **Feature Importance**
Compared most common feature importance methods - Spearman correlation, PCA, drop column, and permutation importance. Demonstrated each one's strengths and weaknesses - for example, that drop column and permutation are model-dependent and that drop column requires retraining whereas permutation does not. Building on this, an automatic feature selection algorithm is included, which drops weak predictors until the loss no longer improves.

###### **Decision Tree**
Using NumPy and object-oriented programming, implemented decision tree classifier and decision tree regressor algorithms. The software fits on training and predicts on a test set. The model finds the best split point in each column that decreases the loss (MSE or Gini Impurity). Once the model is fit, the model predicts by walking decision nodes until a leaf node is reached. The prediction is the average of the leaf for regression and the mode for classification.

###### **Random Forest**
Building on the Decision Tree project, I implemented this ensemble method. The model fits multiple decision trees using bootstrapping (subsampling with replacement) and feature sampling (a random subset of the columns) so that each tree trains on different data. At prediction time, the model makes a weighted average of each tree for regression and the mode of the tree predictions for classification. Out-of-bag error estimates were an optional method.

###### **Gradient Boosting and AdaBoost**
Recreated the boosting algorithm for MSE with fit and predict methods and the AdaBoost algorithm. Written in Python with unit tests for quality checks.

###### **Linear Models**
OLS and Ridge Regression with NumPy. Also implemented Logistic Regression for classification passed through sigmoid and log-likelihood functions. The model uses Gradient Descent to find the correct model weights. The code utilized the object-oriented classes and methods in Python.

###### **Naïve Bayes**
Used the Naïve Bayes theorem to make a classifier algorithm in Python, which predicted whether a movie had a positive or negative review. The model included k-fold cross-validation during training.

## Distributed Computing Projects

###### **Predict Amazon Rating Based on Review Text**:
Used PySpark in a DataBricks environment to preprocess 6.5 million Amazon reviews - including imputation, dropping nulls, and joining, stored in MongoDB. Used Word2Vec, CountVectorizer, and TF-IDF as the final processing step for analysis. Predicted the rating from only review text using PySpark MLlib models - SVM, GBM, RandomForest, Naive Bayes, and Logistic Regression. Achieved 0.72 F1 score and 0.75 Accuracy on the test set.

## Data Acquisition

###### **Search Engine Implementation**
I processed 17 GB of text files using linear, indexed/dictionary, and object-oriented implementation of a hashtable. The three methods were implemented to compare performance. The pipeline rendered the top 100 results in HTML and Jinja.

###### **TF-IDF Document Summarization**
Parsed XML, tokenized, and removed stopwords from all 10K Reuters articles. Used TF-IDF to find the most relevant words in the document/corpus.

###### **Article Recommender**   
This project was hosted on a website using an AWS EC2 instance with Flask. Users found an article they liked, and the site recommended five similar articles based on the euclidean distance of the articles' centroids using GloVe embeddings.

###### **Twitter Sentiment Analysis**
Using the Twitter API as the backend - I made a website that displayed the last 100 tweets from a Twitter profile. Each tweet was color-coded by sentiment score, using a green to red gradient for higher to lower sentiment. Sentiment score was found using the VaderSentiment library, and the website was hosted using an EC2 instance running Flask. 
