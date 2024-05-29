from typing import List, Tuple

import numpy as np

from src.common.constants import PERSONALITY_TYPES
from src.config import COARSE_CLASSIFICATION


def get_personality_class_from_response(response: str) -> str:
    if COARSE_CLASSIFICATION:
        first_present = None
        first_index = len(response)

        for personality_type in PERSONALITY_TYPES:
            index = response.find(personality_type)
            if 0 <= index < first_index:
                first_present = personality_type
                first_index = index
        if first_present is None:
            return ""
        return first_present

    else:
        response = response.lower()
        personality_class = []
        stem_mappings = [
            {"introv": "I", "extrov": "E"},
            {"intuit": "N", "sens": "S"},
            {"think": "T", "feel": "F"},
            {"judg": "J", "perceiv": "P"}
        ]

        for axis_mapping in stem_mappings:
            smallest_index = 99999
            result_letter = None
            for stem, letter in axis_mapping.items():
                try:
                    i = response.index(stem)
                    if smallest_index is None or i < smallest_index:
                        result_letter = letter
                        smallest_index = i
                except ValueError:
                    pass
            if result_letter is not None:
                personality_class.append(result_letter)

        return "".join(personality_class)


def evaluate(predictions: np.array, labels: np.array) -> Tuple[float, List[float]]:
    predictions = np.array([str(x) if not (isinstance(x, float) and np.isnan(x)) else "" for x in predictions])
    global_accuracy = np.mean(predictions == labels)
    global_accuracy *= 100
    print(f"Global accuracy: {global_accuracy:.2f}%")

    axis_accuracies = []
    for i in range(4):
        axis_accuracy = np.mean([(len(prediction) > 0 and prediction[i] == label[i]) for prediction, label in
                                 zip(predictions, labels)]) * 100
        axis_accuracies.append(axis_accuracy)

    return global_accuracy, axis_accuracies
