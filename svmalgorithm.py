import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('phishdataset.csv')

# Use Pandas to encode the string values as numerical values
df = pd.get_dummies(df, columns=['sender', 'subject', 'keywords'])

# Split the data into features and labels
X = df.drop('label', axis=1)  # features
y = df['label']  # labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train the SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate precision
precision = precision_score(y_test, y_pred, pos_label='legitimate')

# Print the classification report
print("Precision:", precision)
print(classification_report(y_test, predictions))
