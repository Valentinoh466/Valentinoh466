import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Collect tweets about Tesla Model Y
tweets = pd.read_csv('tesla_model_y_tweets.csv')

# Step 2: Preprocess the tweets
# Code to preprocess the tweets goes here

# Step 3: Label the tweets
# Code to label the tweets goes here

# Step 4: Split the labeled tweets into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweets['text'], tweets['sentiment'], test_size=0.2, random_state=42)

# Step 5: Vectorize the tweets
# We'll use TfidfVectorizer, but you can also use CountVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train the models
nb_model = MultinomialNB().fit(X_train_vec, y_train)
svm_model = SVC(kernel='linear').fit(X_train_vec, y_train)
lr_model = LogisticRegression().fit(X_train_vec, y_train)

# Step 7: Evaluate the models
nb_preds = nb_model.predict(X_test_vec)
svm_preds = svm_model.predict(X_test_vec)
lr_preds = lr_model.predict(X_test_vec)

print("Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print("Precision:", precision_score(y_test, nb_preds, average='macro'))
print("Recall:", recall_score(y_test, nb_preds, average='macro'))
print("F1-score:", f1_score(y_test, nb_preds, average='macro'))
print()

print("Support Vector Machine:")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print("Precision:", precision_score(y_test, svm_preds, average='macro'))
print("Recall:", recall_score(y_test, svm
