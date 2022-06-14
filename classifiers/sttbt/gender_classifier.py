from classifiers.sttbt.sttbt_classifier import STTBTClassifier


class GenderSTTBTClassifier(STTBTClassifier):

    @classmethod
    def get_model_checkpoint_path(cls) -> str:
        return "classifiers/sttbt/checkpoints/gender_classifier/gender_classifier.pt"
