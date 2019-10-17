import pickle
from sklearn.externals import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
        model = joblib.load(open("tweets.pkl","rb"))

        if request.method == "POST":
            new_tweet = request.form['new_tweet']
            data = [new_tweet]

            data = [word.lower() for word in data]
            data = re.sub(r'(\S*@\S*)|(#\S*)|(\S*\d+\S*)|(https?\S*)|([^a-z ])|(\s?\srt\s\s?)', '', str(data))
            data = re.sub(r'(\s\w\s)|(\s?\srt\s\s?)', ' ', str(data))
            data = data.split()
            list_rt = ["rt"]
            data = [word for word in data if word not in stop_words and word not in list_rt]
            data = [" ".join(data)]

            predicted = model.predict(data)[0]
        return render_template("result.html", prediction = predicted)

if __name__ == '__main__':
        app.run(debug=True)
