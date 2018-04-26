#https://www.youtube.com/watch?v=o_OZdbCzHUA&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=2
#polarity = positiv or negative
#subjectivity = oppinion or fact

import tweepy
from textblob import TextBlob

consumer_key = '4d9CP7abOvzWUE5G97jTyqFMq'
consumer_secret = 'iIqp62yQBCj3liTDfDCFJG1LQFt6etbhgpgFXbAPsd9Y7jnGNq'

access_token = '886443925-Ytjh4QZ0kZagwJgJ1ygZd77FXZkhckPt2CYJSy8n'
access_token_secret = 'k7MaKPz087wpwRzJtWabZU23mDz8mlhZFF0Mh5LVYQ9wi'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('artificial intelligence')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
