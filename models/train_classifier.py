import re
import nltk
import pandas as pd
import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    accuracy_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_path, table_name):
    """
    Loads data from SQL table
    :param: database_path: `str` storing the database file path
    :param: table_name: `str` storing the name of the table to read the SQL data from.
    :return: `pandas.DataFrame` object
    """

    engine = create_engine(f"sqlite:///{database_path}")
    return pd.read_sql_table(table_name, engine)


def get_training_target_data(df):
    """
    Returns training data and target values for the model
    :param: df: `pandas.DataFrame` object
    :return: tuple object containing training data and target values
    """

    X = df['message']
    y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
            'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
            'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
            'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]

    return X, y


def tokenize(text):
    """
    Tokenizes the given text.
    :param: text: `str`
    :return: `list` of tokens
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build a GridSearchCV model
    """

    # build pipeline tokenizing the text, applying TF-IDF feature extractor and finally building a
    # multioutput classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model printing the accuracy, precision and recall for each category
    """

    y_pred = model.predict(X_test)

    for idx, column in enumerate(y_test.columns):
        y_pred_values = y_pred[:, idx]
        y_test_values = y_test[column]

        print(f'Category: {column}')
        print(classification_report(y_test_values, y_pred_values))
        print(f'Accuracy: {accuracy_score(y_test_values, y_pred_values)}')
        print()


def save_model(model, model_save_path):
    """
    Saves model using pickle module
    """
    pickle.dump(model, open(model_save_path, 'wb'))


if __name__ == "__main__":
    args = sys.argv

    if len(args) < 3:
        print("Usage: python train_classifier.py [database_path] [model_save_path]")
        sys.exit()

    db_path, model_save_loc = args[1:3]
    df = load_data(db_path, "MessageCategories")
    X, y = get_training_target_data(df)

    model = build_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_save_loc)
