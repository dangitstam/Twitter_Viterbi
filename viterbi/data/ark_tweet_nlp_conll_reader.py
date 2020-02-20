def read_ark_tweet_conll(file_path):
    # Instances take the form of <token> <tab> <POS> on each line, and instances
    # are separated by a blank line. Read the entire dataset to memory and
    # split by blank lines to collect the instances.

    instances = open(file_path, "r").read().split("\n\n")
    for instance in instances:

        # Skip blank entries (e.g. if there is a newline at EOF).
        if not instance:
            continue

        token_label_pairs = instance.split("\n")
        tokens, labels = zip(*[x.split("\t") for x in token_label_pairs])

        yield list(tokens), list(labels)
