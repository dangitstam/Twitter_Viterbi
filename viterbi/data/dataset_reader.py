from allennlp.data.vocabulary import Vocabulary


class DatasetReader:
    """
    TODO: Docs

    TODO: Include preprocessing here for now (unk'ing, start and stop)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        reader,
        token_namespace: str = "tokens",
        label_namespace: str = "labels",
        token_preprocessing_fn=None,
    ):
        self.vocab = vocab
        self.reader = reader  # TODO: Better name
        self.token_namespace = token_namespace
        self.label_namespace = label_namespace
        self.token_preprocessing_fn = token_preprocessing_fn

    def read(self, file_path):

        # TODO: Make dataset reader it's own class that takes the parser as a function
        # That way a static method isn't being awkwardly tossed around
        instances = self.reader(file_path)
        for tokens, labels in instances:

            if self.token_preprocessing_fn:
                tokens = self.token_preprocessing_fn(tokens)

            token_ids = [
                self.vocab.get_token_index(token, self.token_namespace)
                for token in tokens
            ]
            label_ids = [
                self.vocab.get_token_index(label, self.label_namespace)
                for label in labels
            ]

            yield {
                "tokens": tokens,
                "labels": labels,
                "token_ids": token_ids,
                "label_ids": label_ids,
            }
