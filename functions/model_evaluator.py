# Import necessary libraries and modules
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluation(model, x_test, y_test):

    # Remaking predictions on the test set
    y_pred = model.predict(x_test)

    # Reevaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Storing the results
    print("Accuracy =", accuracy)
    print("Precision =", precision)
    print("Recall =", recall)
    print("F1 Score =", f1)
    print("Confusion Matrix:")
    print(conf_matrix)
