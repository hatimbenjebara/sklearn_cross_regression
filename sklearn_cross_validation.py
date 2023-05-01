from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Load the Boston Housing dataset
boston = load_iris()
X = boston.data
y = boston.target

# Create a linear regression model
model = LinearRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Print the average score and standard deviation
print(f"Average score: {scores.mean()}")
print(f"Standard deviation: {scores.std()}")

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load iris dataset
data = load_iris()
X = data.data
y = data.target

# Initialize a logistic regression model
logreg = LogisticRegression()

# Initialize a KFold object with 5 folds
kf = KFold(n_splits=5)

# Iterate over the folds
scores = []
for train_index, test_index in kf.split(X):
    # Split data into training and testing sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training data for this fold
    logreg.fit(X_train, y_train)

    # Test the model on the testing data for this fold and calculate accuracy
    y_pred = logreg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)

# Calculate the average accuracy across all folds
avg_acc = sum(scores) / len(scores)
print("Average accuracy:", avg_acc)
