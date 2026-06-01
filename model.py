import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from data import prepare_data


def train_model():

    # Load cleaned and prepared data
    data = prepare_data()

    # --- Step 2: Feature Selection ---

    # Features used to predict the target
    features = ["Return", "MA5", "MA20", "Volatility"]

    # X = explanatory variables
    X = data[features]

    # y = target variable
    y = data["Target"]

    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # STEP 3: Model Building

    # Create Random Forest classification model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model using training data
    model.fit(X_train, y_train)

    
    
    # STEP 4: Evaluation

    # Make predictions using test data
    predictions = model.predict(X_test)

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Measure feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, data, accuracy, cm, importance


if __name__ == "__main__":

    _, _, accuracy, cm, importance = train_model()

    print("Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nFeature Importance:")
    print(importance)