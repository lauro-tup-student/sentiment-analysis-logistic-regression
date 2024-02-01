Sentiment Analysis

This project performs sentiment analysis on text data using a logistic regression model. It predicts the emotion associated with a given text comment.

Getting Started

Prerequisites

- Python 3.x
- pandas
- scikit-learn
- Dataset

Dataset
The dataset used for this project is stored in the file dataset.csv. It contains comments and their corresponding emotions. Here are a few examples from the dataset:


Installation

1. Clone the repository:
2. Install the required dependencies:

Usage

1. Prepare the dataset:

   - Create a CSV file containing the text comments and their corresponding emotions. The file should have a column named "Comment" for the text comments and a column named "Emotion" for the corresponding emotions.

2. Update the `input_file` variable in the `sentiment_analysis.py` file with the path to your dataset CSV file.

3. Run the script:
4. Enter a text comment when prompted. The script will predict the emotion associated with the comment and display the result.

Features

- Data cleaning: The script performs lowercase conversion and removes non-alphabetic characters from the text comments.
- Feature engineering: The script uses the TF-IDF vectorization technique to convert the text comments into numerical features.
- Logistic regression model: The script trains a logistic regression model to predict the emotions based on the text features.
