# app.py
from flask import Flask, render_template, request
import pickle
# import webbrowser -> package for browser open
import webbrowser


# Load model
with open('model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")
@app.route("/check", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        email_text = request.form["email_text"]
        input_vec = vectorizer.transform([email_text])
        prediction = model.predict(input_vec)[0]
        result = "🚨 Phishing Email Detected!" if prediction == 1 else "✅ Safe Email."
    return render_template("index.html", result=result)

 
if __name__ == "__main__":
   webbrowser.open("http://127.0.0.1:5000")#--> auto open web browser on program   
   app.run(debug=False)