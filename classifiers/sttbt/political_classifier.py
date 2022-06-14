from classifiers.sttbt.sttbt_classifier import STTBTClassifier


class PoliticalSTTBTClassifier(STTBTClassifier):

    @classmethod
    def get_model_checkpoint_path(cls) -> str:
        return "classifiers/sttbt/checkpoints/political_classifier/political_classifier.pt"
