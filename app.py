from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Example root route (optional but helpful)
@app.route('/')
def home():
    return "Fake News Detection API is Live!"

# Main prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    prediction = model.predict(vectorizer.transform([text]))
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run()