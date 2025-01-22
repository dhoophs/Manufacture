from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

app = Flask(__name__)

# Initialize model and scaler variables
model = None
scaler = None

@app.route('/home', methods=['GET'])
def index():
    return "Flask app is running!", 200

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Debugging: Log the incoming request
        app.logger.info(f"Request.files: {request.files}")
        app.logger.info(f"Request.form: {request.form}")

        # Check if the 'file' key is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        
        # Check if a file was actually selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file to the upload directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        return jsonify({"message": f"File uploaded successfully! Saved at {file_path}"}), 200
    except Exception as e:
        # Catch any unexpected errors
        app.logger.error(f"Error during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Route for training the model
@app.route('/train', methods=['POST'])
def train_model():
    try:
        global model, scaler
        # Load the uploaded dataset
        df = pd.read_csv('synthetic_manufacturing_data.csv')
        
        # Features and target
        X = df[['Temperature', 'Run_Time']]
        y = df['Downtime_Flag']
        
        # Add interaction term
        X['Temp_Run_Interaction'] = X['Temperature'] * X['Run_Time']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = LogisticRegression(class_weight='balanced', C=0.1, solver='liblinear')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Save the model and scaler
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        return jsonify({"message": "Model trained successfully!", "accuracy": accuracy, "f1_score": f1}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model, scaler
        if model is None or scaler is None:
            model = joblib.load('model.pkl')
            scaler = joblib.load('scaler.pkl')
        
        # Parse input JSON
        data = request.get_json()
        if not all(key in data for key in ['Temperature', 'Run_Time']):
            return jsonify({"error": "Invalid input. 'Temperature' and 'Run_Time' are required."}), 400
        
        temperature = data['Temperature']
        run_time = data['Run_Time']
        interaction_term = temperature * run_time
        
        # Prepare the input
        X_new = pd.DataFrame([[temperature, run_time, interaction_term]], columns=['Temperature', 'Run_Time', 'Temp_Run_Interaction'])
        X_scaled = scaler.transform(X_new)
        
        # Predict
        prediction = model.predict(X_scaled)
        confidence = model.predict_proba(X_scaled).max()
        
        return jsonify({"downtime_flag": int(prediction[0]), "confidence_score": round(confidence, 2)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
