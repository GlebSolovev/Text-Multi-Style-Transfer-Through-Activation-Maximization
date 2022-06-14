from typing import Dict, NoReturn

from classifiers.sttbt.political_classifier import PoliticalSTTBTClassifier
from classifiers.sttbt.sentiment_classifier import SentimentSTTBTClassifier
from classifiers.sttbt.gender_classifier import GenderSTTBTClassifier


def classify_and_print_with_sttbt_classifier(classifier_cls, additional_args: Dict, title: str, text: str) -> NoReturn:
    classifier = classifier_cls(batch_size=1, **additional_args)
    batch = classifier.transform_to_batches([text])[0]
    result = classifier.classify(batch)
    print(f"{title}: {result[0]}")


def main():
    text = "democratic men are the most beautiful in the world!"
    additional_args = {"max_text_length_in_tokens": 50, "gpu": False}
    classify_and_print_with_sttbt_classifier(SentimentSTTBTClassifier, additional_args, "sentiment", text)
    classify_and_print_with_sttbt_classifier(GenderSTTBTClassifier, additional_args, "gender", text)
    classify_and_print_with_sttbt_classifier(PoliticalSTTBTClassifier, additional_args, "political", text)


if __name__ == '__main__':
    main()
