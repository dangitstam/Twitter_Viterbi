import numpy as np
from allennlp.data import Vocabulary

from viterbi.models.hidden_markov_model import HiddenMarkovModel


class Model:
    def __init__(
        self,
        order: int,
        viterbi_decoder,
        vocab: Vocabulary,
        start_token: str,
        end_token: str,
        token_namespace: str,
        label_namespace: str,
    ):
        # Collect HMM and Viterbi parameters.
        self.order = order
        self.viterbi_decoder = viterbi_decoder

        self.vocab = vocab
        self.hmm = None
        self.start_token = start_token
        self.end_token = end_token
        self.token_namespace = token_namespace
        self.label_namespace = label_namespace
        self._start_token_id = self.vocab.get_token_index(
            start_token, namespace=label_namespace
        )
        self._end_token_id = self.vocab.get_token_index(
            end_token, namespace=label_namespace
        )
        self.is_trained = False

    def train_model(self, instances: iter):
        # Train a hidden markov model to learn transition and emission probabilities.
        hmm = HiddenMarkovModel(
            self.vocab,
            order=self.order,
            token_namespace=self.token_namespace,
            label_namespace=self.label_namespace,
            start_token=self.start_token,
            end_token=self.end_token,
        )
        hmm.train(instances)

        self.hmm = hmm
        self.is_trained = True

    def predict(self, input_tokens: list):
        input_token_ids = [self.vocab.get_token_index(token) for token in input_tokens]

        output = self.viterbi_decoder(
            input_token_ids,
            self.hmm.emission_matrix,
            self.hmm.transition_matrix,
            self._start_token_id,
            self._end_token_id,
        )

        prediction_label_ids = output.get("label_ids")
        prediction_labels = list(
            map(
                lambda x: self.vocab.get_token_from_index(x, self.label_namespace),
                prediction_label_ids,
            )
        )
        output["labels"] = prediction_labels

        return output

    def evaluate(self, instances):
        # Evaluate model performance on the dev set.
        correctly_tagged_words, total_words = 0, 0
        max_acc, max_acc_example, min_acc, min_acc_example = None, None, None, None
        for instance in instances:
            input_tokens = instance["tokens"]
            labels = instance["labels"]

            output = self.predict(input_tokens)
            correctly_labeled = [
                int(prediction == label)
                for prediction, label in zip(output["labels"], labels)
            ]
            correctly_tagged_words += sum(correctly_labeled)
            total_words += len(labels)

            log_likelihood = self.hmm.log_likelihood(
                instance["token_ids"], output["label_ids"]
            )
            assert np.isclose(log_likelihood, output["log_likelihood"])

            acc = sum(correctly_labeled) / len(labels)
            if min_acc is None or acc < min_acc:
                min_acc = acc
                min_acc_example = (labels, output["labels"])
            elif max_acc is None or acc > max_acc:
                max_acc = acc
                max_acc_example = (labels, output["labels"])

        accuracy = correctly_tagged_words / total_words

        return {
            "accuracy": accuracy,
            "best_example": max_acc_example,
            "worst_example": min_acc_example,
        }
