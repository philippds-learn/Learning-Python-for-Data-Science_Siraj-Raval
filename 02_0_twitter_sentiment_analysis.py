#https://www.youtube.com/watch?v=o_OZdbCzHUA&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=2

from textblob import TextBlob

wiki = TextBlob("testing out textblob")
print(wiki.tags)
print(wiki.words)
print(wiki.polarity)
