from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from common.dataset import generate_splits, load_data
from common.evaluation import evaluate

if __name__ == '__main__':
    data = load_data("/data/datasets/mbti_1.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = generate_splits(data)

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True,
                                 stop_words="english",
                                 )
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    for model in [SVC(kernel='linear'),
                  RandomForestClassifier(),
                  MultinomialNB()]:
        print(f"{model}")
        model.fit(train_vectors, y_train)
        predictions = model.predict(test_vectors)
        evaluate(predictions, y_test)
