import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

tf.get_logger().setLevel('INFO')

df = pd.read_csv("D:/Documents/Sem3AiProject/Chat-master/controller/Tweets.csv")
df.head()
df.columns
tweet_df = df[['text','airline_sentiment']]
tweet_df.head(5)
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
#print(tweet_df.shape)
tweet_df.head(5)
tweet_df["airline_sentiment"].value_counts()
sentiment_label = tweet_df.airline_sentiment.factorize()
#print(sentiment_label)
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)
model = kr.models.load_model('D:/Documents/Sem3AiProject/Chat-master/controller/models')
##history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)
##plt.plot(history.history['accuracy'], label='acc')
##plt.plot(history.history['val_accuracy'], label='val_acc')
##plt.legend()
##plt.show()
##plt.savefig("Accuracy plot.jpg")
##plt.plot(history.history['loss'], label='loss')
##plt.plot(history.history['val_loss'], label='val_loss')
##plt.legend()
##plt.show()
##plt.savefig("Loss plot.jpg")

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
##    print(model.predict(tw))
##    print("Predicted label: ", sentiment_label[1][prediction])
    label = sentiment_label[1][prediction]
    if label == "positive":
        print("This message is safe.")
    else:
        print("There is a high chance that this message is a threat/fraud.")

##test_sentence1 = "It was good"
##predict_sentiment(test_sentence1)
##test_sentence2 = "Remember me, I need help , Transfer money immediately"
##predict_sentiment(test_sentence2)
