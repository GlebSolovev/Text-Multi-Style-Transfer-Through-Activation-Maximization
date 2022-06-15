from hpay.sentiment_transformer import SentimentHPAYTransformer
from formality_styleformer import FormalityStyleformerTransformer


def main():
    text = "This film is very boring!"
    print(f"Initial: {text}")

    sentiment_transformer = SentimentHPAYTransformer(to_positive=True)
    positive = sentiment_transformer.transfer([text])[0]
    print(f"More positive: {positive}")

    formality_transformer = FormalityStyleformerTransformer(to_formal=True)
    formalized = formality_transformer.transfer(positive)
    print(f"And then formalized: {formalized}")


if __name__ == '__main__':
    main()
