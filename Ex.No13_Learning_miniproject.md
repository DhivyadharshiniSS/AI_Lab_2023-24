# Ex.No: 10 Learning – Use Supervised Learning  
### DATE:04-11-2024                                                                            
### REGISTER NUMBER :212222060051
### AIM: 
To write a program to train the classifier for Spam classification.
###  Algorithm:
Algorithm for Spam Classifier
Load and Preprocess Data:
Load the dataset (SMS Spam Collection).
Convert the labels to binary (spam = 1, ham = 0).
Data Splitting:
Split the dataset into training (80%) and testing (20%) sets.
Text Vectorization:
Convert text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
Train the Classifier:
Use Naive Bayes as the classifier (MultinomialNB) to train the model.
Model Evaluation:
Evaluate the model on the test set and print the accuracy and classification report.
Test with Custom Messages:
Use the trained model to predict if custom messages are spam or ham.
Display the result for each message.

### Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and Preprocess the Data
# Load the SMS Spam Collection dataset (URL or local file)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Step 2: Label Encoding - Convert labels to binary values (1 for spam, 0 for ham)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Step 4: Text Vectorization - Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Test with custom messages
sample_messages = [
    "Congratulations! You've won a $1000 gift card!",
    "Hey, are we still on for lunch tomorrow?",
    ]

# Correctly transform the sample messages using the same vectorizer
sample_tfidf = vectorizer.transform(sample_messages)
predictions = model.predict(sample_tfidf)

# Display results for sample messages
for message, pred in zip(sample_messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Message: '{message}'\nPrediction: {label}\n")
```

### Output:
```
Accuracy: 0.98

Classification Report:
               precision    recall  f1-score   support

           0       0.98      1.00      0.99       966
           1       0.99      0.89      0.94       149

    accuracy                           0.98      1115
   macro avg       0.99      0.95      0.97      1115
weighted avg       0.98      0.98      0.98      1115

Message: 'Congratulations! You've won a $1000 gift card!'
Prediction: Spam

Message: 'Hey, are we still on for lunch tomorrow?'
Prediction: Ham

```
### Result:
Thus the system was trained successfully and the prediction was carried out.
