import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the saved model."
    )
    args = parser.parse_args()

    archived_model = pickle.load(open(args.model_path, "rb"))
    model = archived_model.get("model")
    dataset_reader = archived_model.get("dataset_reader")

    interact = input("\nModel trained. Test with a few examples? (y/n): ")
    if interact.lower() == "y":
        try:
            while True:
                tweet = input("\nType an example tweet (press CTRL-C to exit): ")
                tokens = tweet.split()
                processed_tokens = dataset_reader.token_preprocessing_fn(tokens)
                output = model.predict(processed_tokens)
                prediction_labels = output["labels"]
                print("Input: {}".format(tweet))
                print("Output:")
                print("Prediction: {}".format(" ".join(prediction_labels)))
                print(
                    "\n".join(
                        [
                            "Token: {:25s} Label: {:25s}".format(token, label)
                            for token, label in zip(tokens, prediction_labels)
                        ]
                    )
                )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
