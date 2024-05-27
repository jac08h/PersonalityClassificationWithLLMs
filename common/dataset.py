import re
from typing import Optional

from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from common.constants import INSTRUCTION_TEMPLATE, LETTERS_TO_WORDS, PERSONALITY_TYPES, \
    RESPONSE_TEMPLATE
from config import COARSE_CLASSIFICATION, REMOVE_LINKS, REMOVE_PERSONALITY_TYPES


def load_data(data_path):
    data = pd.read_csv(data_path)
    data["posts"] = data["posts"].apply(
        lambda x: preprocess_text(x)
    )
    return data


def preprocess_text(text):
    if REMOVE_LINKS:
        text = re.sub(r'https?://(?:www\.)?([^/\s]+)[^\s]*',
                      lambda m: f"[{m.group(1).upper().replace('.', '_')}_LINK]",
                      text)
    if REMOVE_PERSONALITY_TYPES:
        text = remove_personality_type(text)

    return text


def generate_splits(data, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        data['posts'], data['type'], test_size=test_size, stratify=data['type'], random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def format_datapoint(text: str, personality_class: Optional[str] = None) -> str:
    personality_class = personality_class if personality_class else ""

    if COARSE_CLASSIFICATION:
        return f"""{INSTRUCTION_TEMPLATE}Person's forum posts separated by |||: {text}
    ### Personality classification (options: ISTJ, ISFJ, INFJ, INTJ, ISTP, ISFP, INFP, INTP, ESTP, ESFP, ENFP, ENTP, ESTJ, ESFJ, ENFJ, ENTJ).
    First letter: Introversion (I) – Extroversion (E)
    Second letter: Intuition (N) – Sensing (S)
    Third letter: Thinking (T) – Feeling (F)
    Fourth letter: Judging (J) – Perceiving (P)

    From these options, the person can be best classified as {RESPONSE_TEMPLATE}{personality_class}"""

    else:
        personality_class = ", ".join([LETTERS_TO_WORDS[letter] for letter in personality_class])
        return f"""{INSTRUCTION_TEMPLATE}Person's forum posts separated by |||: {text}
        ### Personality classification.
        First axis: Introvert – Extrovert
        Second axis: Intuition – Sensing
        Third axis: Thinking – Feeling
        Fourth axis: Judging – Perceiving

        On each of these axis (separated by commas) the person can be best classified as {RESPONSE_TEMPLATE}{personality_class}"""


def remove_personality_type(text: str, placeholder: str = "[redacted]") -> str:
    for personality_type in PERSONALITY_TYPES:
        text = text.replace(personality_type, placeholder)
        text = text.replace(personality_type.lower(), placeholder)
    return text


def create_dataset(X, y, max_size=None) -> Dataset:
    data = []
    for text, personality_class in zip(X, y):
        splits = text.split("|||")
        if len(splits) < 10:
            data.append(format_datapoint(text, personality_class))
            continue

        quarter_length = len(splits) // 4

        first_part = "|||".join(splits[:quarter_length])
        second_part = "|||".join(splits[quarter_length: 2 * quarter_length])
        third_part = "|||".join(splits[2 * quarter_length: 3 * quarter_length])
        fourth_part = "|||".join(splits[3 * quarter_length:])

        data += [format_datapoint(first_part, personality_class),
                 format_datapoint(second_part, personality_class),
                 format_datapoint(third_part, personality_class),
                 format_datapoint(fourth_part, personality_class)]

    if max_size is not None:
        data = data[:max_size]

    df = pd.DataFrame(data, columns=['text'])
    return Dataset.from_pandas(df)
