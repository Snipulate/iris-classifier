import os
import argparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    joblib.dump(model, "outputs/iris_model.joblib")

if __name__ == "__main__":
    main()
