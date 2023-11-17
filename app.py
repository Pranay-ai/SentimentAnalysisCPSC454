print("Importing Libraries")
from flask import Flask,render_template,request
import requests
import pickle
import re
from multiprocessing import context
import nltk
import pickle
import numpy
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
aws_acess_key_id = 'Your Acess Key'
aws_secret_key="Your Secret Key"
api_link = 'https://9v8y7h8huh.execute-api.us-west-2.amazonaws.com/stage-1/api-sentiment'
print("Loading Libraries")
nltk.download('stopwords')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''

    sentence = sen.lower()

    # Remove html tags
    sentence = re.compile(r'<[^>]+>').sub('',sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence
def get_response(input_text):
    input_text = preprocess_text(input_text)
    input_text = tokenizer.texts_to_sequences([input_text])
    input_text = pad_sequences(input_text, padding='post', maxlen=100)
    input_text = input_text.tolist()
    data = {'body':input_text}
    response = requests.post(api_link, json=data, auth=(aws_acess_key_id, aws_secret_key))
    return response.json()['body']
    

app=Flask(__name__)
@app.route("/")
def Index():

    return render_template("Index.html")

@app.route('/', methods=['POST'])
def home():
    text=request.form['textData']
    response = get_response(text)
    score=float(response[1])
    print(response)
    if score<0.4:
        string = "              The sentiment of the text is negative               "
    elif score>0.4 and score<0.6:
        string = "              The sentiment of the text is neutral                "
    else:
        string = "              The sentiment of the text is  positive              "


    return render_template('Index.html',result=string)


    




if __name__=="__main__":
    app.debug=True
    app.run()