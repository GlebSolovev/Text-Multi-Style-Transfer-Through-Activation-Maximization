from classifiers.sttbt.sttbt_classifier import STTBTClassifier


class SentimentSTTBTClassifier(STTBTClassifier):

    @classmethod
    def get_model_checkpoint_path(cls) -> str:
        return "classifiers/sttbt/checkpoints/sentiment_classifier/sentiment_classifier.pt"
