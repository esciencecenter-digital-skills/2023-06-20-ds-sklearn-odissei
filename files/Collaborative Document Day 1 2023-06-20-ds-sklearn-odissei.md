![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1 2023-06-20-ds-sklearn-odissei

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/odissei-ml-1

Collaborative Document day 1: https://tinyurl.com/odissei-ml-1

Collaborative Document day 2: https://tinyurl.com/odissei-ml-2

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## Conclusions from the pre-workshop survey
* Most of you program quite often, but not in Python
* You want to learn about Python
* You want to learn about machine learning
* You want to learn about CBS microdata --> Next week (but you will learn how to deal with the data today and tomorrow)
* You have diverse backgrounds: So: rule no. 1: Ask for help/clarification, especially if it seems other people ask more advanced questions. There are no stupid questions!

## 🖥 Workshop website

https://esciencecenter-digital-skills.github.io/2023-06-20-ds-sklearn-odissei/

🛠 Setup

https://esciencecenter-digital-skills.github.io/2023-06-20-ds-sklearn-odissei/#setup


## 👩‍🏫👩‍💻🎓 Instructors

Sven van der Burg, Djura Smits
## 🧑‍🙋 Helpers

Ji Qi, Carsten Schnober

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda
09:00	Welcome and icebreaker
09:15	Introduction to Machine Learning
10:00	Break
10:10	Predictive modeling pipeline: data exploration
11:00	Break
11:10	Predictive modeling pipeline: Fitting a scikit-learn model on numerical data
12:00	Lunch Break
13:00   Introduction to LISS data (Joris Mulder)
13:10	Predictive modeling pipeline: Fitting a scikit-learn model on numerical data
14:00	Break
14:10	Predictive modeling pipeline: Handling categorical data
15:00	Break
15:10	Predictive modeling pipeline: Handling categorical data
15:45	Wrap-up
16:00	END

## 🏢 Location logistics
* Coffee and toilets?
* In case of an emergency?
* **Wifi**: eduroam or Text “summer2023” To: +316 3525 0006

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises

### Exercise: Data exploration (15min)

Imagine we are interested in predicting penguins species based on two of their body measurements: culmen length and culmen depth. First we want to do some data exploration to get a feel for the data.

The data is located in `datasets/penguins_classification.csv`.

Load the data with Python and try to answer the following questions:

1.   How many features are numerical? How many features are categorical?
2.   What are the different penguins species available in the dataset and how many samples of each species are there?
3.    Plot histograms for the numerical features
4.    Plot features distribution for each class (Hint: use seaborn.pairplot).
5.    Looking at the distributions you got, how hard do you think it will be to classify the penguins only using “culmen depth” and “culmen length”?

1, 2:
```python
import pandas as pd

penguins = pd.read_csv("penguins_classification.csv")
penguins

penguins["Species"].value_counts()
```
3:
```python
penguins.hist()
```
4:
```python
import seaborn as sns

sns.pairplot(penguins, hue="Species")
```
5:
A statistical model can perform well, but will not be perfect due to overlaps


### 📝 Exercise: Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score` API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```

#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the data and target loaded above
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.



### Exercise: Compare with simple baselines
The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with LogisticRegression) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.

What would be the score of a model that always predicts ' >50K'?

What would be the score of a model that always predicts ' <=50K'?

Is 81% or 82% accuracy a good score for this problem?

Use a DummyClassifier such that the resulting classifier will always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

You can import DummyClassifier like this:
```python
from sklearn.dummy import DummyClassifier
```



## 🧠 Collaborative Notes

### Setup

Installation:
```shell
git clone https://github.com/INRIA/scikit-learn-mooc
[...]
cd scikit-learn-mooc
conda env create -f environment.yml
# This takes a few minutes 
Collecting package metadata (repodata.json): done
Solving environment: done
[...]
```
Check your installation:
```shell
conda activate scikit-learn-course
python check_env.py 

Using python in [...]/opt/anaconda3/envs/scikit-learn-course
3.11.4 | packaged by conda-forge | (main, Jun 10 2023, 18:10:28) [Clang 15.0.7 ]

[ OK ] numpy version 1.25.0
[ OK ] scipy version 1.10.1
[ OK ] matplotlib version 3.7.1
[ OK ] sklearn version 1.2.2
[ OK ] pandas version 2.0.2
[ OK ] seaborn version 0.12.2
[ OK ] notebook version 6.5.4
[ OK ] plotly version 5.15.0
```
Run Jupyter notebooks
```shell
conda activate scikit-learn-course
cd scikit-learn-mooc  # (if not done above already)
jupyter lab
```


Check your current working directory in a Notebook:
```python
pwd
```

### Data Exploration

```python
import pandas as pd

adult_census = pd.read_csv("datasets/adult-census.csv")

adult_census.head()

target_column = "class"
adult_census[target_column].value_counts()
```

```python
numerical_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

all_columns = numerical_columns + categorical_columns + [target_column]
adult_census = adult_census[all_columns]
```

```python
print(
    f"The dataset contains {adult_census.shape[0]} samples and "
    f"{adult_census.shape[1]} columns"
)
```

```python
adult_census.shape
```

```python
print(f"The dataset contains {adult_census.shape[1] - 1} features")
```

```python
adult_census.hist(figsize=(20, 14))
```

```python
adult_census["sex"].value_counts()
```

```python
adult_census["education"].value_counts()
```

```python
pd.crosstab(
    index=adult_census["education"], columns=adult_census["education-num"]
)
```

```python
import seaborn as sns
```

```python
n_samples_to_plot = 5000

columns = ["age", "education-num",  "hours-per-week"]

sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={
        "alpha": 0.2
    },
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)
```

### 2. Predicting model with numerical data

First, start with loading the data file and have a look at the top 5 rows there.

```python=
import pands ad pd
adult_census = pd.read_csv("path-to-your-csv-file")
adult_census.head()
```
Then we want to have the target column in a separate variable.

```python=
target_name = "class"
target = adult_census(target_name)
target.head()
```

And get the remaining data frame (without target).

```python=
data = adult_census.drop(columns=[traget_name])
data.head()
```

To know the shape of the data

```python=
data.shape
```

Note that `shape` is an attribute of data, while `head()` is method/function, those are different.

#### First model

Now we are ready to train our first model that is the K-nearest neighbors model. If you want to know more details regarding this algorithm, have a look at its [wikipedia page](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) and [sklearn page](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

We will use `sklearn`, the Python package for machine learning. 

```python=
from sklearn.neighbors import KNeighborsClassfier

model = KNeighborsClassifier() # initialize the model instance
model.fit(data, target) # method to train the model
```

A good thing there is that sklearn provides sensible defaults of the hyperparameters of the models, and you can check that by running the code below in your jupyter notebook.

```python=
?KNeighborsClassifier
```

Learning can be represented like this:
![](https://codimd.carpentries.org/uploads/upload_135b364a30bb008142896540cded4729.png)

Now, let's do predict!

```python=
target_predicted = model.predict(data)
```

And this is how the model do prediction:
![](https://codimd.carpentries.org/uploads/upload_68c8d69f0a86de8245747201015d2f91.png)

To have a look at the predictions
```python=
target_predicted[:5], target[:5]
```

This gives the first 5 predictions and the ground truth target labels correspondingly. You can also run the follows to have the accuracy of prediction in general (think about why).

```python=
(target == target_predicted).mean()
```

#### Train-test split

Note that we are testing the "memory" here, but we instead want to test the "generalization", and we need a skill called train-test split for that purpose. Here is how to make it:

```python=
adult_census_test = pd.read_csv("path-to-your-test-file")
target_test = adult_census_test(target_name)
data_test = adult_census_test.drop(columns=[target_test])
```

To find out the accuracy of the model, we do

```python=
accuracy = model.score(data_test, target_test)
accuracy
```

And here is how the model is scored:
![](https://codimd.carpentries.org/uploads/upload_0383053626212a357c3872c3e9e2aee5.png)

Till now, we finish our first machine learning cycle that includes `fit`, `predict`, and `score`.

#### Working with numerical data

Again, start from loading the dataset:

```python=
adult_census = pd.read_csv("path-to-your-data-file")
adult_census = adult_census.drop(columns="education-num")
adult_census.head()
```
and we separate our data to have features and target

```python=
data = adult_census.drop(columns="class")
target = adult_census["class"]
```

Now, to have a look at the data types of the features to know who are numerical:

```python=
data.dtypes
```

To get the numerical columns:

```python=
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]
```

Now, to make a train-test split:

```python=
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data_numeric, target, random_state=42, test_size=0.25) # fix random_state for reproduction
data_train.shape, data_test.shape
```

Note that you may choose different train-test ratios, depending on how many data points you have in your dataset.

The next thing is to train a Logistic Regression model. Check the [sklearn page](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for more details of this algorithm. Here is the code:

```python=
from sklearn.linear_model import LogisticRegression

model = logisticRegression()
model.fit(data_train, target_train)
```

To evaluate the model:

```python=
model.score(data_test, target_test)
```

## :notebook: Feedback morning session
Write down one thing that went well and one thing we can improve. Think about: pace of the workshop, content, setup instructions, the help you got, how we teach, exercises, etc. Any feedback is useful!
### What went well?:
* The baseline of this is good. Getting hand-on experience is very important. I am going to steal this setup for some of my own workshops.
* I think it's great!
* Very interactive and helpful introduction. The pace is great for all levels.
* I really liked it! The speed was good. And thanks for allowing time for questions.
* After setting all up, everything went rather smooth!
* Coding and running the code went good. 
* Good pace, everything was quite clear, instructors and helpers are being very nice
* Good setup with the notebooks 
* I think the pace is good. The material as well
* Very clear and intuitive tutorial, had fun!
* Very helpful to do this with multiple instructors that can help.
* The pace was good, and the instructions were very clear, guiding use through (for someone who doesn't have much prior experience)
* Thank you for guiding us through. The pace was good and I think nobody was left behind. 
* I liked that we learned the basics together and then did something similar on our own, so that we had a starting place but also needed to learn how to do a couple of new things during the penguin exercise. 

### What can be improved?:
* Nothing much. Having taught programming myself, the intro class is always a bit tricky.
* Maybe the discussion of the excercise answers could be more interactive. Ask the participants what problems they encouter.
* Could you give a brief overview about different types of Python elements, so it is easier for us to manipulate and wrangle the data?
* I think it would be helpful to first give an overview of the different data elements (lists, dataframes) and skip the part on qualitative/quantitative variables whose relevance is not very clear 
* The setup process was a bit chaotic, but it also went well at the end. 
* The setup. Otherwise it seems fine so far.
* Maybe explain a little bit more about the 'notebook' setup VS a 'run from top the bottom script'.
* Was very nice, sometimes it was a bit too quick to catch up with the commands. And maybe a bit more explanation on the code themselves (even just very shortly)
* Too many platforms (Jupyter tutorial, GitHub, carpentries) can be confusing sometimes
* The prior installation is a bit confusing
* Just the prework, I could have installed everything ahead of time and we would have had more time :)

* Finding the initial website was a bit tough (not sure if I missed something in slack) - <-- Yes, good point. It would be great if there would just be 'core links' section in the collaborative document.
* All was good for me. Maybe explain ahead of time that the Penguin task will be to classify species. 
* Overall I really liked it! Just a minor thing: maybe you can explain "basic" things such as how to use the terminal, how to use anaconda, etc.
* I think you can be more explicit about what you would like people do up front, it would make sense to just say: complete datacamp introduction + intermediary.

### Sven's summary:
- Some python basics:
```python
# Importing a package into our Python session
import pandas as pd

# Assigning a variable
x = 2

# Creating a list
my_list = [1, 2, 3]

# Accessing 'something'
my_list[0] # first element in the list
data['Species'] # a column in a pandas dataframe
```
- The tools we use:
    - Jupyter lab (similar to jupyter notebook) -> Useful for experimenting, because of its interactiveness. You can also type Python in a script and run it from there. Also supports other programming languages.
    - Anaconda --> Python package manager, allows you to manage different Python environments with all the different versions of packages installed
    - Terminal
- To minimize chaos on your laptop: You only need to look at jupyter lab and the screen. In principle only look at the collaborative document when you do exercises.
- We will try to be better at making sure setup is done before the workshop :)

## Wrapping up:
- Access to LISS data
- If you have to miss tomorrow. Finish the 'predictive modeling pipeline' module from https://inria.github.io/scikit-learn-mooc/predictive_modeling_pipeline/predictive_modeling_module_intro.html'. Work on https://esciencecenter-digital-skills.github.io/fertility-prediction-assignment/. Read through tomorrow's collaborative document (after tomorrow): https://tinyurl.com/odissei-ml-2
- Tomorrow:
    - start at 9:00, you can be there a bit earlier if you have questions/problems
    - we will do continue with the live coding programme, then you can work on the LISS data
## :notebook: Feedback afternoon session
Write down one thing that went well and one thing we can improve. Think about: pace of the workshop, content, setup instructions, the help you got, how we teach, exercises, etc. Any feedback is useful!
### What went well?:
- I enjoyed the live coding: quick explanation, and then we can try it out
- Live coding was very fun and interactive! The fact that there are 4 teachers really helps everyone to get to know ML, thank you very much!
-
- Getting into actual coding and machine learning was very cool!
- Live coding was nice and easy to follow, very cool that we already got to do our first (very simple) machine learning programs
- Good amount of breaks, good plan of the schedual
- the pace was overall good, and easy to follow. Also nice to get exposure to a variety of approaches.
### What can be improved?:
- 
-
- Overall I think we go a bit fast, but its understandable as we are behind schedual.
- It might be nice to occasionally mention where things might go wrong (since so far it went very smoothly) as things to watch out for when working with panel data, but I realize this could also lead to more confusion, so maybe in a separated brief part before we start each exercise.
-
-

### Sven's summary:
- You like the live coding, yay!
- Pace is good for most of you, we will try to go a bit slower
- We realize we are working on toy problems, and we're showing the happy flow. It will get more 'dirty' when working with the LISS data. In the meantime, if something goes wrong with your code please share, so we can trouble shoot together!

## 📚 Resources

- Git for Windows: https://gitforwindows.org/
- eScience Center Trainings & Workshops: https://www.esciencecenter.nl/digital-skills/
- Fairlearn: https://fairlearn.org/
- Scikit-Learn API Reference: https://scikit-learn.org/stable/modules/classes.html
- Scikit-Learn User Guide: https://scikit-learn.org/stable/user_guide.html
- LISS data: https://eyra.co/benchmark/5/15