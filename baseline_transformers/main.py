from formality_styleformer import FormalityStyleformerTransformer


def main():
    text = "It's piece of cake, we can do it"
    print(f"Initial: {text}")

    formality_transformer = FormalityStyleformerTransformer(to_formal=True)
    formalized = formality_transformer.transfer(text)
    print(f"Formalized: {formalized}")


if __name__ == '__main__':
    main()
