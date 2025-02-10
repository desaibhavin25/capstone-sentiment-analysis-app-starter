from flask import Flask, jsonify, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)


def sentiment_analysis(input):
            user_sequences = tokenizer.texts_to_sequences([input])
            user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
            prediction = model.predict(user_sequences_matrix)
            return round(float(prediction[0][0]),2)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_dict = []
    sentimentText = 'Neutral:'

    def load_keras_model():
        global model
        model = load_model('models/uci_sentimentanalysis.h5')

    def load_tokenizer():
        global tokenizer
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

    with app.app_context():
        load_keras_model()
        load_tokenizer()
   
    text = request.form.get("user_text");
    if text is not None and text != "":
       
        sid = SentimentIntensityAnalyzer()
        sentiment_dict = sid.polarity_scores(text)
        sentiment_dict["Custom"] = sentiment_analysis(text)


        '''print("Overall sentiment dictionary is : ", sentiment_dict)
        print("Sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
        print("Sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
        print("Sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

        print("Sentence Overall Rated As", end=" ")'''

        # Decide sentiment as positive, negative, or neutral
        if sentiment_dict['compound'] >= 0.05 :
             sentimentText = 'Positive:'
        elif sentiment_dict['compound'] <= -0.05 :
            sentimentText =  'Negative:'
        
    return render_template('form.html', sentiment = sentiment_dict, sentimentText = sentimentText)
    

if __name__ == "__main__":
 app.run(port=4000)