class Predictor:
    def __init__(
        self,
        vocab,
        hidden_markov_model,
        viterbi_decoder,
        start_token,
        end_token,
        token_namespace,
        label_namespace,
    ):
        self.vocab = vocab
        self.hmm = hidden_markov_model
        self.viterbi_decoder = viterbi_decoder
        self.start_token = start_token
        self.end_token = end_token

    def __call__(self, input_tokens):
        output = self.viterbi_decoder(
            input_tokens,
            self.hmm.emission_matrix,
            self.hmm.transition_matrix,
            self.vocab.get_token_index(self.start_token, self.label_namespace),
            self.vocab.get_token_index(self.end_token, self.label_namespace),
        )

        prediction_labels = [
            self.vocab.get_token_from_index(label_id)
            for label_id in output["label_ids"]
        ]

        output["prediction_labels"] = prediction_labels

        return output
