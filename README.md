![Medium Blog](https://medium.com/@cheruiyotemmanuel/starbucks-capstone-challenge-9ff9e0d20ede)

![Starbucks logo](https://th.bing.com/th/id/OIP.IRO-Tiax2ckaruqxocO__QHaHa?w=172&h=180&c=7&r=0&o=5&pid=1.7)

<h1><center> Starbuck's Capstone Challenge </center><h1>

## Table of Contents
1. Problem
2. Results
3. Explanation
4. Dependencies
5. References

This is a project to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer, in a Starbuck simulated dataset.

## Problem 
Aim: Predict if a customer will respond to an offer.

Approach & Metrics:
- Merge offer, customer and transaction data.
- Evaluate accuracy and F1-score of a naive model that assumes all offers successful.
- Compare logistic regression, random forest and gradient boosting models.
- Refine the best model based on accuracy and F1-score.

## Results
Model ranking based on accuracy:
- RandomForestClassifier: 0.742
- GradientBoostingClassifier: 0.736
- LogisticRegression: 0.722
- Naive model: 0.471

Model ranking based on F1-score:
- RandomForestClassifier: 0.735
- GradientBoostingClassifier: 0.725
- LogisticRegression: 0.716
- Naive model: 0.640

**Random Forest performed best in terms of accuracy and F1-score.**

## Explanation
- Logistic regression constructs a linear boundary to separate successful and unsuccessful offers, but non-linear boundary expected based on customer demographics.
- Random forest and gradient boosting are ensemble methods combining multiple decision trees.
- Random forest combines trees by majority voting, gradient boosting reduces misclassified samples iteratively.
- These strategies affect the depth and randomness of decision trees, which affects bias and variance.

## Dependencies
- ProgressBar Library ``` pip install progressbar2 ```
- Python 3.5+
- Machine Learning Libraries: NumPy,SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

## References
- Model accuracy definition: https://developers.google.com/machine-learning/crash-course/classification/accuracy
- F1-score definition: https://developers.google.com/machine-learning/crash-course/classification/f1-score
- Assessment of models with imbalanced classes: https://towardsdatascience.com/assessing-the-performance-of-machine-learning-models-on-imbalanced-datasets-76b50f75eb0b
- Extending beyond accuracy, precision, and recall: https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
- Comprehensive overview of logistic regression: https://towardsdatascience.com/a-comprehensive-guide-to-logistic-regression-2d056f2b9b3a
- Random forest algorithm introduction: https://towardsdatascience.com/random-forest-algorithm-for-beginners-93fceab87234
- Gradient boosting algorithm overview: https://towardsdatascience.com/an-introduction-to-gradient-boosting-bb724ac24bb8
- Multi label binarizer function: https://scikit-learn
