# Author Pankhuri Mishra
#from nltk.tokenize import word_tokenize
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.corpus import wordnet
#from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
#import re
#import string
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-logistic_regression.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        print("Loading self.vectorizer")
        #print(self.vectorizer)
        #print(type(self.vectorizer))
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        print("Loading self.user_final_rating")
        print(self.user_final_rating)
        print("Number of columns:")
        print(len(self.user_final_rating.columns))
        print("columns:")
        print(self.user_final_rating.columns)
        self.data = pd.read_csv("dataset/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))
        print("Loading self.cleaned_data")
        print(self.cleaned_data)
        print("Number of columns:")
        print(len(self.cleaned_data.columns))
        print("columns:")
        print(self.cleaned_data.columns)
        #self.lemmatizer = WordNetLemmatizer()
        #print("Lemmatization")
        #print(lemmatizer)
        #self.stop_words = set(stopwords.words('english'))
        #print("stop words")
        #print(stop_words)

    """function to get the top product 20 recommendations for the user"""

    def getRecommendationByUser(self, user):
        recommedations = []
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""

    def getSentimentRecommendations(self, user):
        if user not in self.user_final_rating.index:
            print(f"The User {user} does not exist. Please provide a valid user name")
            return None
        else:
            # Get top 20 recommended products from the best recommendation model
            print("A**",self.user_final_rating.loc[user])
            print("B**", self.user_final_rating.loc[user].sort_values(ascending=False)[0:20])
            print("C**", self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            top20_recommended_products = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            # Get only the recommended products from the prepared dataframe "df_sent"
            df_top20_products = self.cleaned_data[self.cleaned_data.name.isin(top20_recommended_products)]
            # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
            print("Top 20 products")
            print(df_top20_products)
            print("Values")
            print(df_top20_products["reviews_lemmatized"].values)
            print("Values as string")
            print(df_top20_products["reviews_lemmatized"].values.astype(str))
            X = self.vectorizer.transform(df_top20_products["reviews_lemmatized"].values.astype(str))
            # Use the best sentiment model to predict the sentiment for these user reviews
            df_top20_products['predicted_sentiment'] = self.model.predict(X)
            print("df_top20_products")
            print(df_top20_products['predicted_sentiment'])
            # Create a new column to map Positive sentiment to 1 and Negative sentiment to 0. This will allow us to easily summarize the data
            df_top20_products['positive_sentiment'] = df_top20_products['predicted_sentiment'].apply(
                lambda x: 1 if x == 1 else 0)
            # Create a new dataframe "pred_df" to store the count of positive user sentiments
            pred_df = df_top20_products.groupby(by='name').sum()
            ##pred_df.columns = ['pos_sent_count']
            # Create a column to measure the total sentiment count
            pred_df['total_sent_count'] = df_top20_products.groupby(by='name')['predicted_sentiment'].count()
            # Create a column that measures the % of positive user sentiment for each product review
            pred_df['post_sent_percentage'] = np.round(pred_df["positive_sentiment"] / pred_df['total_sent_count'] * 100, 2)
            # Return top 5 recommended products to the user
            result = list(pred_df.sort_values(by='post_sent_percentage', ascending=False)[:5].index)
            return result

    """function to classify the sentiment to 1/0 - positive or negative - using the trained ML model"""

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    """function to preprocess the text before it's sent to ML model"""

    def preprocess_text(self, text):

        # cleaning the review text (lower, removing punctuation, numericals, whitespaces)
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans('', '', string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text

    """function to get the pos tag to derive the lemma form"""

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    """function to remove the stop words from the text"""

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    """function to derive the base lemma form of the text using the pos tag"""

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(
            self.remove_stopword(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(
            tag[1])) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)