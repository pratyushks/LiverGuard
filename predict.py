import pandas as pd
import pickle

# Function to load user inputs from a CSV file
def load_user_inputs(csv_file):
    """
    Load user inputs from a CSV file.
    Assumes the columns in the CSV are in the same order as required by the model.
    Converts 'Male' in the Gender column to 1 and others to 0.
    """
    try:
        # Load CSV file into a DataFrame
        user_inputs = pd.read_csv(csv_file)
        print(f"Loaded {len(user_inputs)} records from {csv_file}.")

        # Convert 'Gender' column: 'Male' to 1, others to 0
        if 'Gender' in user_inputs.columns:
            user_inputs['Gender'] = user_inputs['Gender'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
            print("Converted 'Gender' column to binary (1 for Male, 0 for others).")

        return user_inputs
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Function to predict using the model
def predict_from_csv(input_df, model_file):
    """
    Predict outcomes using a trained model and user input DataFrame.
    """
    try:
        # Load the trained model
        with open(model_file, 'rb') as file:
            ensemble_model = pickle.load(file)

        # Predict using the model
        predictions = ensemble_model.predict(input_df)

        # Map predictions to human-readable output
        prediction_output = {
            0: "No symptoms found for liver cancer",
            1: "You may have liver cancer, consult a doctor"
        }
        results = [prediction_output[pred] for pred in predictions]

        # Add predictions to the input DataFrame
        input_df['Prediction'] = results
        print("Predictions completed successfully.")
        return input_df
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # Specify the CSV file and model file
    csv_file = "user_inputs.csv"
    model_file = "models/predictions.pkl"

    # Load user inputs from the CSV file
    user_inputs_df = load_user_inputs(csv_file)

    if user_inputs_df is not None:
        # Predict using the loaded DataFrame
        predictions_df = predict_from_csv(user_inputs_df, model_file)

        # Save results to a new CSV file
        if predictions_df is not None:
            output_file = "predictions_output.csv"
            predictions_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}.")
