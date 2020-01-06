import numpy as np


def viterbi(
    input_tokens, emission_matrix, transition_matrix, start_token_id, end_token_id,
):
    """
    Docs
    """
    order = len(transition_matrix.shape)
    label_namespace_size = transition_matrix.shape[0]
    num_tokens = len(input_tokens)

    # Likelihoods will be computed in log space to avoid underflow.
    # pylint: disable=assignment-from-no-return
    emission_matrix = np.log2(emission_matrix)
    transition_matrix = np.log2(transition_matrix)

    # Initial path starting at position zero with only the start token begins
    # with probability 1 to prevent any paths not prefixed with start tokens from
    # being an optimal path. Note: log(1) = 0.
    paths = np.zeros([num_tokens + 1] + [label_namespace_size] * (order - 1),)
    init_path_index = [0] + [start_token_id] * (order - 1)
    paths[tuple(init_path_index)] = 0

    backpointers = np.zeros(
        [num_tokens + 1] + [label_namespace_size] * (order - 1), dtype=int
    )

    for k, token in enumerate(input_tokens, start=1):
        # Shape: (K ^ order).
        r = (
            transition_matrix
            + np.expand_dims(paths[k - 1], axis=-1)
            + emission_matrix[token]
        )

        # Update backpointers.
        backpointers[k] = r.argmax(axis=0)

        # Update paths.
        paths[k] = np.amax(r, axis=0)

    # Explain this lol
    stop_transition = (transition_matrix.T)[end_token_id].T
    r_final = paths[num_tokens] + stop_transition

    # Backtrace
    output_indices = list(np.unravel_index(np.argmax(r_final), r_final.shape))
    for k in range(num_tokens - 2, 0, -1):
        yk = backpointers[k + 2][output_indices[0]][output_indices[1]]
        output_indices.insert(0, yk)

    # Collect the log likelihood of the path chosen by viterbi.
    log_likelihood = np.max(r_final)

    return {
        "log_likelihood": log_likelihood,
        "label_ids": output_indices
    }


def trigram_viterbi(
    input_tokens, emission_matrix, transition_matrix, start_token_id, end_token_id,
):
    """
    Docs
    """
    order = len(transition_matrix.shape)
    label_namespace_size = transition_matrix.shape[0]
    num_tokens = len(input_tokens)

    # Likelihoods will be computed in log space to avoid underflow.
    # pylint: disable=assignment-from-no-return
    emission_matrix = np.log2(emission_matrix)
    transition_matrix = np.log2(transition_matrix)

    # Initial path starting at position zero with only the start token begins
    # with probability 1 to prevent any paths not prefixed with start tokens from
    # being an optimal path. Note: log(1) = 0.
    paths = np.zeros([num_tokens + 1] + [label_namespace_size] * (order - 1),)
    init_path_index = [0] + [start_token_id] * (order - 1)
    paths[tuple(init_path_index)] = 0

    backpointers = np.zeros(
        [num_tokens + 1] + [label_namespace_size] * (order - 1), dtype=int
    )

    for k, token in enumerate(input_tokens, start=1):
        for u in range(transition_matrix.shape[0]):
            for v in range(transition_matrix.shape[0]):

                # For each ngram suffixed with u, v, find a prefixing w
                # that maximizes 
                max_r, max_r_index = float('-inf'), -1
                for w in range(transition_matrix.shape[0]):
                    r = paths[k - 1][w][u] + transition_matrix[w][u][v] + emission_matrix[token][v]
                    if r > max_r:
                        max_r = r
                        max_r_index = w

                # Update backpointers.
                backpointers[k][u][v] = max_r_index

                # Update paths.
                paths[k][u][v] = max_r

    # Explain this lol
    penultimate_token, final_token = None, None
    max_final = float('-inf')
    for u in range(transition_matrix.shape[0]):
        for v in range(transition_matrix.shape[0]):
            r_final = paths[-1][u][v] + transition_matrix[u][v][end_token_id]
            if r_final > max_final:
                max_final = r_final
                penultimate_token = u
                final_token = v

    # Backtrace
    output_indices = [penultimate_token, final_token]
    for k in range(num_tokens - 2, 0, -1):
        yk = backpointers[k + 2][output_indices[0]][output_indices[1]]
        output_indices.insert(0, yk)

    # Collect the log likelihood of the path chosen by viterbi.
    log_likelihood = max_final

    return {
        "log_likelihood": log_likelihood,
        "label_ids": output_indices
    }
