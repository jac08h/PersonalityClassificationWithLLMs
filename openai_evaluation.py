import pandas as pd

from common.classifiers import OpenAILMClassifier
from common.dataset import generate_splits, load_data
from common.evaluation import get_personality_class_from_response


if __name__ == '__main__':
    openai_lm_classifier = OpenAILMClassifier("davinci-002")
    data = load_data("/data/datasets/mbti_1.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_splits(data)

    responses = openai_lm_classifier.predict(X_test)
    predictions = [get_personality_class_from_response(response) for response in responses]
    df = pd.DataFrame({"prediction": predictions, "response": responses})
    df.to_csv("results/openai_lm_predictions.csv", index=False)