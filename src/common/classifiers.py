from abc import abstractmethod
from random import choices
from typing import List

import openai
from tqdm import tqdm

from src.common.constants import PERSONALITY_TYPES
from src.config import DEVICE
from dataset import format_datapoint


class Classifier:
    @abstractmethod
    def train(self, data, labels) -> None:
        pass

    def predict(self, data, **kwargs) -> List[str]:
        pass


class RandomClassifier(Classifier):
    def __init__(self):
        self.proportion_by_class = {}
        super().__init__()

    def train(self, data, labels) -> None:
        # get proportions of each class
        value_counts = labels.value_counts()
        for label in value_counts.index:
            self.proportion_by_class[label] = value_counts[label] / len(labels)

    def predict(self, data, **kwargs) -> List[str]:
        weights = [self.proportion_by_class[label] for label in PERSONALITY_TYPES]
        return choices(PERSONALITY_TYPES, weights=weights, k=len(data))


class LocalLMClassifier(Classifier):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        super().__init__()

    def train(self, data, labels):
        pass

    def predict(self, data, batch_size, **kwargs) -> List[str]:
        responses = []
        for i in tqdm(range(0, len(data), batch_size), desc="Processing",
                      total=len(data) // batch_size + (len(data) % batch_size > 0)):
            batch_data = data[i:i + batch_size]
            prompts = [format_datapoint(text) for text in batch_data]

            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
            outputs = self.model.generate(**input_ids, **kwargs)

            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(output[len(input_ids["input_ids"][j]):], skip_special_tokens=True)
                responses.append(response)
        return responses


class OpenAILMClassifier(Classifier):
    def __init__(self, model_name: str):
        self.client = openai.OpenAI()
        self.model_name = model_name
        super().__init__()

    def predict(self, data, **kwargs) -> List[str]:
        responses = []
        for text in tqdm(data, desc="Processing"):
            prompt = format_datapoint(text)
            response_object = self.client.completions.create(model=self.model_name,
                                                             prompt=prompt,
                                                             max_tokens=20,
                                                             temperature=0)
            responses.append(response_object.choices[0].text.strip())

        return responses
