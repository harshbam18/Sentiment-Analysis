from warnings import filterwarnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: '%.2f' % x)
df = pd.read_csv("C:\Users\Harsh Sharma\Desktop\Sentiment_Analysis_on_Amazon_Product_Reviews-main\amazon_review.csv", sep=",")
df.head()
def text_preprocessing(dataframe, dependent_var):
  # Normalizing Case Folding - Uppercase to Lowercase
  dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

  # Removing Punctuation
  dataframe[dependent_var] = dataframe[dependent_var].str.replace('[^\w\s]','')

  # Removing Numbers
  dataframe[dependent_var] = dataframe[dependent_var].str.replace('\d','')

  # StopWords
  sw = stopwords.words('english')
  dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

  # Remove Rare Words
  temp_df = pd.Series(' '.join(dataframe[dependent_var]).split()).value_counts()
  drops = temp_df[temp_df <= 1]
  dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

  # Lemmatize
  dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

  return dataframe
df = text_preprocessing(df, "reviewText")
df["reviewText"].head()
def text_visulaization(dataframe, dependent_var, barplot=True, wordcloud=True):
  # Calculation of Term Frequencies
  tf = dataframe[dependent_var].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
  tf.columns = ["words", "tf"]

  if barplot:
    # Bar Plot
    tf[tf["tf"]>1000].plot.barh(x="words", y="tf")
    plt.title("Calculation of Term Frequencies : barplot")
    plt.show()

  if wordcloud:
    # WordCloud
    text = " ".join(i for i in dataframe[dependent_var])
    wordcloud = WordCloud(max_font_size=100, max_words=1000, background_color="white").generate(text)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Calculation of Term Frequencies : wordcloud")
    plt.show()
    wordcloud.to_file("wordcloud.png")
    text_visulaization(df, "reviewText")
    def create_polarity_scores(dataframe, dependent_var):
        sia = SentimentIntensityAnalyzer()
        dataframe["polarity_score"] = dataframe[dependent_var].apply(lambda x: sia.polarity_scores(x)["compound"])
    create_polarity_scores(df, "reviewText")
    df.head()
    def create_label(dataframe, dependent_var, independent_var):
        sia = SentimentIntensityAnalyzer()
        dataframe[independent_var] = dataframe[dependent_var].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
        dataframe[independent_var] = LabelEncoder().fit_transform(dataframe[independent_var])

        X = dataframe[dependent_var]
        y = dataframe[independent_var]
        return X, y
    X, y = create_label(df, "reviewText", "sentiment_label")
    # Split Dataset
    def split_dataset(dataframe, X, y):
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=1)
        return train_x, test_x, train_y, test_y
    train_x, test_x, train_y, test_y = split_dataset(df, X, y)
    def create_features_count(train_x, test_x):
        # Count Vectors
        vectorizer = CountVectorizer()
        x_train_count_vectorizer = vectorizer.fit_transform(train_x)
        x_test_count_vectorizer = vectorizer.fit_transform(test_x)
        return x_train_count_vectorizer, x_test_count_vectorizer
    x_train_count_vectorizer, x_test_count_vectorizer = create_features_count(train_x, test_x)
    def create_features_TFIDF_word(train_x, test_x):
  # TF-IDF word
        tf_idf_word_vectorizer = TfidfVectorizer()
        x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
        x_test_tf_idf_word = tf_idf_word_vectorizer.fit_transform(test_x)
        return x_train_tf_idf_word, x_test_tf_idf_word
    x_train_tf_idf_word, x_test_tf_idf_word = create_features_TFIDF_word(train_x, test_x)
    def create_features_TFIDF_ngram(train_x, test_x):
        tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,3))
        x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(train_x)
        x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(test_x)
        return x_train_tf_idf_ngram, x_test_tf_idf_ngram
    x_train_tf_idf_ngram, x_test_tf_idf_ngram = create_features_TFIDF_ngram(train_x, test_x)
    def create_features_TFIDF_chars(train_x, test_x):
        tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,3))
        x_train_tf_idf_chars = tf_idf_chars_vectorizer.fit_transform(train_x)
        x_test_tf_idf_chars = tf_idf_chars_vectorizer.fit_transform(test_x)
        return x_train_tf_idf_chars, x_test_tf_idf_chars
    x_train_tf_idf_chars, x_test_tf_idf_chars = create_features_TFIDF_chars(train_x, test_x)
    # Random Forest
def crate_model_randomforest(train_x, test_x):
  # Count
    x_train_count_vectorizer, x_test_count_vectorizer = create_features_count(train_x, test_x)
    rf_count = RandomForestClassifier()
    rf_model_count = rf_count.fit(x_train_count_vectorizer, train_y)
    accuracy_count = cross_val_score(rf_model_count, x_test_count_vectorizer, test_y, cv=10).mean()
    print("Accuracy - Count Vectors: %.3f" % accuracy_count)

    # TF-IDF Word
    x_train_tf_idf_word, x_test_tf_idf_word = create_features_TFIDF_word(train_x, test_x)
    rf_word = RandomForestClassifier()
    rf_model_word = rf_word.fit(x_train_tf_idf_word, train_y)
    accuracy_word = cross_val_score(rf_model_word, x_test_tf_idf_word, test_y, cv=10).mean()
    print("Accuracy - TF-IDF Word: %.3f" % accuracy_word)

  # TF-IDF ngram
    x_train_tf_idf_ngram, x_test_tf_idf_ngram = create_features_TFIDF_ngram(train_x, test_x)
    rf_ngram = RandomForestClassifier()
    rf_model_ngram = rf_ngram.fit(x_train_tf_idf_ngram, train_y)
    accuracy_ngram = cross_val_score(rf_model_ngram, x_test_tf_idf_ngram, test_y, cv=10).mean()
    print("Accuracy TF-IDF ngram: %.3f" % accuracy_ngram)

  # TF-IDF chars

    rf_chars = RandomForestClassifier()
    rf_model_chars = rf_chars.fit(x_train_tf_idf_chars, train_y)
    accuracy_chars = cross_val_score(rf_model_chars, x_test_tf_idf_chars, test_y, cv=10).mean()
    print("Accuracy TF-IDF Characters: %.3f" % accuracy_chars)

    return rf_model_count, rf_model_word, rf_model_ngram, rf_model_chars
rf_model_count, rf_model_word, rf_model_ngram, rf_model_chars = crate_model_randomforest(train_x, test_x)
def model_tuning_randomforest(train_x, test_x):
  # Count
    x_train_count_vectorizer, x_test_count_vectorizer = create_features_count(train_x, test_x)
    rf_model_count = RandomForestClassifier(random_state=1)
    rf_params = {"max_depth": [2,5,8, None],
               "max_features": [2,5,8, "auto"],
               "n_estimators": [100,500,1000],
               "min_samples_split": [2,5,10]}
    rf_best_grid = GridSearchCV(rf_model_count, rf_params, cv=10, n_jobs=-1, verbose=False).fit(x_train_count_vectorizer, train_y)
    rf_model_count_final = rf_model_count.set_params(**rf_best_grid.best_params_, random_state=1).fit(x_train_count_vectorizer, train_y)
    accuracy_count = cross_val_score(rf_model_count_final, x_test_count_vectorizer, test_y, cv=10).mean()
    print("Accuracy - Count Vectors: %.3f" % accuracy_count)

    return rf_model_count_final
rf_model_count_final = model_tuning_randomforest(train_x, test_x)
def predict_count(train_x, model, new_comment):
  new_comment= pd.Series(new_comment)
  new_comment = CountVectorizer().fit(train_x).transform(new_comment)
  result = model.predict(new_comment)
  if result==1:
    print("Comment is Pozitive")
  else:
    print("Comment is Negative")
predict_count(train_x, model=rf_model_count, new_comment="this product is very bad :)")
new_comment=pd.Series(df["reviewText"].sample(1).values)
new_comment
predict_count(train_x, model=rf_model_count, new_comment=new_comment)