from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train_linear_regression(Xtrain, Ytrain):
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    return model

def train_logistic_regression(Xtrain, Ytrain, C=10):
    model = LogisticRegression(C=C, max_iter=1000)
    for i in tqdm(range(1000), desc="Training Logistic Regression"):
        model.fit(Xtrain, Ytrain > 0)
    return model

def train_random_forest(Xtrain, Ytrain):
    model = RandomForestClassifier(random_state=2, n_estimators=100)
    for i in tqdm(range(100), desc="Training Random Forest"):
        model.fit(Xtrain, Ytrain > 0)
    return model

def evaluate_regression_model(model, Xtrain, Ytrain, Xtest, Ytest):
    train_score = model.score(Xtrain, Ytrain)
    test_score = model.score(Xtest, Ytest)
    return train_score, test_score

def evaluate_classification_model(model, Xtrain, Ytrain, Xtest, Ytest):
    train_score = accuracy_score(Ytrain, model.predict(Xtrain))
    test_score = accuracy_score(Ytest, model.predict(Xtest))
    return train_score, test_score

def predict(model, Xtrain, Xtest):
    Ptrain = model.predict(Xtrain)
    Ptest = model.predict(Xtest)
    return Ptrain, Ptest
