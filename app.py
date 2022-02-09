from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from os.path import dirname, join, realpath
import re
from string import punctuation
import emoji
import joblib
import csv
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from wordcloud import STOPWORDS

if nltk.find('corpora/omw-1.4') == False:
    nltk.download('omw-1.4')

app = Flask(__name__)

CORS(app, origins=["http://127.0.0.1:3000"])

with open(
    join(dirname(realpath(__file__)), "models/tweet_classifier_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)

@app.route('/')
def index():
    return "Hello World!"


@app.route('/get-tweet', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_tweet():
    with open('./data/cyberbullyng_update.csv') as f:
        reader = csv.reader(f)
        chosen_row = random.choice(list(reader))
        print(chosen_row[0])
    return {"tweet": chosen_row[0]}

labels = {0 :"not cyberbullying",1: "gender",2 :"ethnicity",3 :"religion",4 :"age"}
def tweet_cleaning(text):
   
    text = text.lower()
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", " ", text)
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would",text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = re.sub(emoji.get_emoji_regexp(),"",text)
    text = re.sub(r'[^\x00-\x7f]','',text)
    
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word) for word in text .split()])
    text = [lemmatizer.lemmatize(word) for word in text .split() if not word in set(STOPWORDS)]
    text = ' '.join(text )
    
    # Return a list of words
    return(text)

@app.route('/cyberbullying', methods=['POST'])
@cross_origin(supports_credentials=True)
def cyber_bullying_prediction():
    jsondata = request.get_json()
    tweet = jsondata['tweet']
    tweet = tweet_cleaning(tweet)
    prediction = model.predict([tweet])
    print(prediction)
    return { 'result': int(prediction[0]), "label":  labels[int(prediction[0])] }

if __name__ == "__main__":
    app.run(debug=True)