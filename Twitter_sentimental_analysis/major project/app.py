from flask import Flask, render_template, redirect, url_for, request
import joblib

# Load the SVM model and CountVectorizer object from disk
filename = 'svm.joblib'
classifier = joblib.load(open(filename, 'rb'))
cv = joblib.load(open('bow_vectorizer.joblib', 'rb'))

app = Flask(__name__, template_folder='Templates')  # Specify the template folder explicitly
def home():
    return render_template('Main.html')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        return redirect(url_for('predict'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)

        # Convert prediction to labels
        if my_prediction[0] == 0:
            prediction_label = "bad comment"
            prediction_label = "bad comment"
            prediction_label = "bad comment"
        else:
            prediction_label = "good comment"

        return render_template('result.html', prediction=prediction_label)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)2255