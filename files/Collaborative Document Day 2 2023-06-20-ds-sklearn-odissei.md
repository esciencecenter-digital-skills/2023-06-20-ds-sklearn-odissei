![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2 2023-06-20-ds-sklearn-odissei

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/odissei-ml-2

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
09:00	Welcome and recap
09:15	Predictive modeling pipeline: working with numerical data
10:00	Break
10:10	Predictive modeling pipeline: Handling categorical data
11:00	Break
11:10	Fertility prediciton using LISS data
12:00	Lunch Break
13:00	Fertility prediciton using LISS data
14:00	Break
14:10	Intermezzo: what steps to take next?
15:00	Break
15:10	Submit results to benchmark
15:45	Recap + Post-workshop Survey
16:00	END

## 🏢 Location logistics
* Coffee and toilets?
* In case of an emergency?
* **Wifi**: ?

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises
Fertility prediction assignment: https://esciencecenter-digital-skills.github.io/fertility-prediction-assignment/

### Exercise: Recap fitting a scikit-learn model on numerical data
#### 1. Why do we need two sets: a train set and a test set?

a) to train the model faster
b) to validate the model on unseen data
c) to improve the accuracy of the model

Select all answers that apply

#### 2. The generalization performance of a scikit-learn model can be evaluated by:

a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function
b) calling fit to train the model on the training set and score to compute the score on the test set
c) calling cross_validate by passing the model, the data and the target
d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply

#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

a) X and y are internally split five times with non-overlapping test sets
b) estimator.fit is called 5 times on the full X and y
c) estimator.fit is called 5 times, each time on a different training set
d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets

Select all answers that apply

#### 4. (optional) Scaling
We define a 2-dimensional dataset represented graphically as follows:
![](https://i.imgur.com/muvSbI6.png)

Question

If we process the dataset using a StandardScaler with the default parameters, which of the following results do you expect:

![](https://i.imgur.com/t5mTlVG.png)


a) Preprocessing A
b) Preprocessing B
c) Preprocessing C
d) Preprocessing D

Select a single answer

#### 5. (optional) Cross-validation allows us to:

a) train the model faster
b) measure the generalization performance of the model
c) reach better generalization performance
d) estimate the variability of the generalization score

Select all answers that apply

#### Answers


### Handling categorical data: encoding of categorical variables

#### Ordinal encoding (everyone gives their answers in the collaborative doc):

- Q1: Is ordinal encoding appropriate for marital status? For which (other) categories in the adult census would it be appropriate? Why?
- Q2: Can you think of another example of categorical data that is ordinal?
- Q3: What problem arises if we use ordinal encoding on a sizing chart with options: XS, S, M, L, XL, XXL? (HINT: explore `ordinal_encoder.categories_`)
- Q4: How could you solve this problem? (Look in documentation of OrdinalEncoder)
- Q5: Can you think of an ordinally encoded variable that would not have this issue?

#### Answers


## 🧠 Collaborative Notes

### 1. Preprocessing numerical features

We start from restarting the kernal of the jupyter notebook and running all cells.

Now, to have an overview of the training data:

```python=
data_train.describe()
```

From there, you may see features that have different ranges of values, and that might be a problem in many cases, e.g., linear and distance-based models are sensitive to value scales of features.

Let's give it a try:
```python=
from sklearn.preprocessing import StandardScaler

scaler = StandardSaler()
scaler = fit(data_train)
```

To have a look at the standarized training data:
```python=
data_train_scaled = scaler.transform(data_train)
data_train_scaled
```

You can also do fitting and transforming together like this:
```python=
scaler.fit_Transform(data_train)
```

With the scaler, we combine it with the ML model by doing:

```python=
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
```

As before, we train the model like this:

```python=
model.fit(data_train, target_train)
```

The process above can be understood like this:
![](https://codimd.carpentries.org/uploads/upload_65e60a69cd7498ead646791ef8ff8d61.png)

To make prediction, we do:

```python=
model.predict(data_test)
```

and to understand this process, it looks like this:
![](https://codimd.carpentries.org/uploads/upload_25f09cb4abdffc22ca9ce0c031247fea.png)

Now, to evaluate the model performance, we do something similar as before:

```python=
model.score(data_test, target_test)
```

### 2. Cross validation

The goal is to find out and evaluate the general performance of models. 

Back to the notebook, this is what we do:
```python=
from sklearn.model_selection import cross_validate

cv_result = cross_validate(model, data_numeric, target, cv=5)
cv_result
```

### 3. Encoding of categorical variables

Since we can't pass strings to ML models, we do encoding on categorical variables. We first select categorical columns.

```python=
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns
```

This wll automatically select the columns whose dtype is object. Then we have the categorical features by doing:

```python=
data_categorical = data[categorical_columns]
data_categorical
```

Now we implement the ordinal encoder:

```python=
from sklearn.preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]
encoder = OrdinalEncoder().set_output(transform="pandas")
encoder.fit_transform(education_column)
```

But in this way, we introduce some order of the values there, which doesn't have to be the order of the original values even. There are encoding techniques without order assumption, and one of them is one-hot encoding.

```python=
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas") # in practice you can ignore the parameters there, it's just for showcase purpose.
encoder.fit_transform(education_column)
```

Now let's try to train a model with encoded data:

```python=
model = make_pipeline(OneHotEncoder(), LogisticRegression(max_iter=500)) # increase max_iter to get convergency of all models, otherwise report error
cross_validate(model, data_categorical, target)
```

But you can still get an error, and by reading the error message you know that one value of the "native-country" column is really rare to appear in all the tests. You can check it out by running:

```python=
data_categorical["native-country"].value_counts()
```

To solve it, do the follows in the above block instead:

```python=
model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), logisticRegression(max_iter=500))
```

Now, to wrap things up and build a more complex pipeline

```python=
from sklearn.compose import ColumnTransformer

encoder = OneHotEncoder(handel_unknown="ignore")
scaler = StandardScaler()
preprocessor = ColumnTransformer([
    ("one-hot-encoder", encoder, categorical_columns), 
    ("standard_scaler", scaler, numerical_columns)
])
```

You can understand the preprocessor in this way:
![](https://codimd.carpentries.org/uploads/upload_3319c2bc14217eb7ad4899922f8b28bc.png)

The pipeline is then defined and validated by doing:

```python=
model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
cv_result = cross_validation(model, data, target, cv=5)
cv_result["test_score"].mean(), cv_result["test_score"].std()
```

### Example code to put in `predict_outcomes` function in `script.py`
```python
    def predict_outcomes(df):
        """Process the input data and write the predictions."""

        # The predict_outcomes function accepts a Pandas DataFrame as an argument
        # and returns a new DataFrame with two columns: nomem_encr and
        # prediction. The nomem_encr column in the new DataFrame replicates the
        # corresponding column from the input DataFrame. The prediction
        # column contains predictions for each corresponding nomem_encr. Each
        # prediction is represented as a binary value: '0' indicates that the
        # individual did not have a child during 2020-2022, while '1' implies that
        # they did.

        # Add your method here instead of the line below, which is just a dummy example.
        import pandas as pd
        data = pd.read_csv('path-to-data.csv') # you know this doesn't work but replace it with what works for you
        target = pd.read_csv('path-to-target.csv')

        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy='most_frequent')
        model.fit(data, target)

        df["prediction"] = model.predict(df)

        return df[["nomem_encr", "prediction"]]
```


## :notebook: Feedback morning session

### What went well?
- Learned a lot! Liked the coding and the time for querstions. 
- Pace was great, good explanation of quite complicated processes. I 
- I really liked that this session built on the session yesterday!
- The mix of teaching and self-testing was a great structure. 
- Great pictures! Also like that we go through the assignments and discuss the correct answers.

### What can be improved?
- I think more attention should be given to feature importance. Without knowing how to do this I think the challenge will be difficult.
- Maybe pay a little more attention to meaningfull semantics, e.g. 'encoder_categorical' instead of 'encoder', also curious about machine learning naming conventions in general.
- I was confused about where to access the data at first. Also, we didn't talk about feature selection. Will we talk about that later? 
- I'm just confused all round. But maybe that's just me. 
- The only thing that could be improved the next time is to have even more time, so that we do not need to leave out any content. 
- Maybe some more attention could be given to any errors we could run into (but I understand there is little time). 
- When we do something wrong or run into an error we correct the code in the cell and continue. From a learning perspective I think it would be useful to keep the wrong code, perhaps as a commented out cell, and then write the correct version below. Then it's easy to learn from the mistakes.

## Sven's summary:
- If you have any questions about the material or if you think something was missing, please ask about it and we can discuss it! For example if you want to dive into feature importance, let's explore!
- We will talk about feature selection in the intermezzo
- Good idea to keep the erroneous code inside cell, commented out, I will use that for next editions!
- We assumed you already had access to the data, but we will communicate it better next time


## 📚 Resources
* Fertility prediction assignment: https://esciencecenter-digital-skills.github.io/fertility-prediction-assignment/
* Link to the data challenge: https://eyra.co/benchmark/5/22

### Resources from intermezzo: what to try next?
* More on evaluation metrics: https://inria.github.io/scikit-learn-mooc/evaluation/evaluation_module_intro.html
* Different machine learning models:
    * In general: Scikit-learns amazing user guide: https://scikit-learn.org/stable/user_guide.html
    * Understanding linear models: https://inria.github.io/scikit-learn-mooc/linear_models/linear_models_module_intro.html
    * Understanding decision tree models: https://inria.github.io/scikit-learn-mooc/trees/trees_module_intro.html
    * Understanding ensemble learning: https://inria.github.io/scikit-learn-mooc/ensemble/ensemble_module_intro.html
    * Support Vector Machines: https://scikit-learn.org/stable/modules/svm.html
    * Introduction to deep learning course (good for understanding DL and creating your own neural networks with Keras): https://carpentries-incubator.github.io/deep-learning-intro/
    * Fast.ai (Python library similar to sklearn, but applying state-of-the-art deep learning): https://docs.fast.ai/
* Hyperparameter tuning: https://inria.github.io/scikit-learn-mooc/tuning/parameter_tuning_module_intro.html
* Feature selection: https://inria.github.io/scikit-learn-mooc/feature_selection/feature_selection_module_intro.html
* Feature importance: https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html
* How to participant the challenge: https://github.com/eyra/fertility-prediction-challenge

## Post-workshop survey
https://www.surveymonkey.com/r/TTFP72W
