# DataScientist Capstone Repository

# Project Name

Sparkify-Project
Data Science Capstone - Churn Prediction at Sparkify
Project Definition
## Overview

The project "Sparkify" is part of Data Science Nanodegree from Udacity. This is the final project where, I have done Churn prediction by creating a machine learning model.
### What we will learn from this project

1. Manipulate large and realistic datasets with Spark to engineer relevant features for predicting churn.
2. Use Spark MLlib to build machine learning models with large datasets.
3. What could be done with non-distributed technologies like scikit-learn.
## List of possible event types are

--------------------------

| Cancel |

| Submit Downgrade|

| Thumbs Down|

| Home |

| Downgrade |

| Roll Advert |

| Logout |

| Save Settings |

| Cancellation Confirmation |

| About |

| Settings |

| Add to Playlist |

| Add Friend |

| NextSong |

| Thumbs Up |

| Help |

| Upgrade |

| Error |

| Submit Upgrade |

--------------------------

#### Where the type 'Cancellation Confirmation' is set equal to users churning.

## Problem

The project is high intensity because of the large amount of data, all of our data wrangling and model creation has to be done using Apache Spark. This adds another level of abstraction on top of the already well-established process. But it does enable us to work with large data than we would otherwise be able to.

## Expected Results

A model for churn prediction have been created and evaluated. The model have been trained and tested on a subset of the 12GB of data, and the final testing should happen on completely separate validation set. An accuracy, F1-Score confusion matrix will be used to evaluate the performance and feasibility of the model.

## Actual Results

End of this project, two main iterations on a churn-prediction model were implemented and evaluated:
1. Model used a simple pivot of the event that seemed to contain the most relevant difference between churning.
2. Non-churning users.

Three models were trained with this data, given the following F1 Values:

1. Gradient Boosted Trees - 68.9%

2. Random Forest - 68.4%

3. Support Vector Machine - 68.4%


While the values look promising, inspection of the confusion matrices revealed that SVM and Random Forest classified all users as non-churning, but because of the relative low number of churned users, the F1 score was still fairly high.

Best performing model for the first iteration was Gradient Boosted Trees.

Second iteration, two new features were introduced. Number of sessions and days from first to last recorded event.

Using these features, the Gradient Boosted Trees algorithm was once again trained.

The results is much better than the initial attempt. With F1-score of 89.1% for validation data, and 294 correctly identified churners, the second iteration of the model is great first model which could be fine-tuned and improved even more.

## Overview of Files

## Data Files

README.md - This readme

Sparkify.html/ipynb - notebook used for practice in Udacity Workspace

mini_sparkify_event_data.json - starting data file containing the streaming music provider's user logs


## Improvement

To achieve the optimal user experience, using more capable hardware and moving the text extraction process from the cloud to the device would be essential. This would reduce the processing time and give access to the outputs of all of the modules of the text extraction pipeline, which would, in turn, enable the following features:

❖ User-guided reading (e.g. read big text first, or read the text the user is pointing at)

❖ Better support for languages other than English

❖ Output filtering (e.g. ignore text smaller than some adjustable threshold)

❖ Passive text detection (auditory cue on text detection, perhaps with additional information encoded in the tone and volume)

The user experience could also be improved significantly by using MXNet, which is a deep learning library that is better optimized for mobile devices than TensorFlow. The speedup wouldn’t be enough for running text extraction on the device, but it would reduce the classification delay significantly.

### Blog Post Link
https://medium.com/@sidheswar.ghosh/churn-prediction-7b95a2ecbaa5



## References

In this project, the course material at Udacity has been used for reference. On top of that, the official pySpark documentation has also been used "https://spark.apache.org/docs/latest/api/python/index.html"
