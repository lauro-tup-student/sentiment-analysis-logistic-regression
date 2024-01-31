import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

input_file = 'C:\\Users\\lauro\\Documents\\learning-purposes\\dataset.csv'

try:

    # load and clean data
    df = pd.read_csv(input_file)
    df['Comment'] = df['Comment'].str.lower().str.replace("[^a-zA-Z ]", "")

    # feature engineering
    vectorized = TfidfVectorizer(stop_words="english", max_features=1000, max_df=0.7)
    features = vectorized.fit_transform(df['Comment'])

    # train model
    model = LogisticRegression()
    model.fit(features, df['Emotion'])


    def predict_emotion(description):
        vectorized_description = vectorized.transform([description])
        prediction = model.predict(vectorized_description)[0]
        print(f"Predicted emotion: {prediction}")


    comment = 'im grabbing a minute to post i feel greedy wrong'
    predict_emotion(comment)


except FileNotFoundError as e:
    print(f"File not found: {e}")
