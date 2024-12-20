import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Load the dataset and preprocess it the same way as in train.py
def load_test_data():
    data = pd.read_csv("data/processed/cleaned_dataset.csv")
    
    # Define features and target variable
    X = data[['year','km_driven','fuel','transmission','owner','engine','max_power']]
    y = data['selling_price']  # Ensure `y` is a single column (target variable)
    
    # Split the data using the same random state
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Apply the same scaling as in train.py
    scaler = StandardScaler()
    X_test_scaled = pd.DataFrame(
        scaler.fit_transform(X_test), 
        columns=X.columns  # Preserve column names for compatibility
    )
    
    return X_test_scaled, y_test

# Load model from file
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    return joblib.load(model_path)

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Compute MAE as the evaluation metric
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    return mae

# Replace current model if the new one is better
def replace_model_if_better(new_model_path, current_model_path, X_test, y_test):
    # Load models
    new_model = load_model(new_model_path)
    current_model = load_model(current_model_path)

    # Evaluate models
    print("Evaluating new model...")
    new_model_score = evaluate_model(new_model, X_test, y_test)
    print("Evaluating current model...")
    current_model_score = evaluate_model(current_model, X_test, y_test)

    # Compare and replace if necessary (lower MAE is better)
    if new_model_score < current_model_score:
        print("The new model is better. Replacing the current model.")
        
        os.rename(new_model_path, current_model_path)  # Rename new model
    else:
        print("The current model is better. .")
         

def main():
    current_model_path = "models/current_model.pkl"
    new_model_path = "models/new_model.pkl"
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Compare and potentially replace the current model
    replace_model_if_better(new_model_path, current_model_path, X_test, y_test)

if __name__ == "__main__":
    main()
