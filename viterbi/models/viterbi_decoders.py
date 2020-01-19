import numpy as np


def viterbi(
    input_tokens, emission_matrix, transition_matrix, start_token_id, end_token_id,
):
    """
    Given a sequence of tokens, along with a set of transition and emission probabilities, outputs
    a dictionary containing the most likely hidden sequence and its log-likelihood.

    Parameters
    ----------
    input_tokens : List[Int]
        A list of ints representing an encoded passage. Note that any entry in this list
        must be at least 0 and at most equal to any of ``transition_matrix``'s dimensions
        (all of which should be identical, see below).
    emission_matrix : np.ndarray
        A matrix such that emission_matrix[x][y] yields the likelihood of label y omitting
        the token x.
    transition_matrix : np.ndarray
        A matrix such that any value at indices [w, u, ..., v] represent the transition
        likelihood from [w, u, ....] to v. Note that all dimensions should be equal.
    start_token_id : int
        The int representing the special START token. This encoding must be consistent with
        the vocabulary with which emission_matrix and transition_matrix are trained.
    end_token_id : int
        The int representing the special START token. This encoding must be consistent with
        the vocabulary with which emission_matrix and transition_matrix are trained.

    Returns
    -------
    A dictionary of the form
    { 'log_likelihood' : The log-likelihood of the most likely sequence under the model.
       'label_ids' : A list of ints containing the predictions for the hidden sequence. }
    """
    order = len(transition_matrix.shape)
    label_namespace_size = transition_matrix.shape[0]
    num_tokens = len(input_tokens)

    # Computing in Log-space
    # ----------------------
    # Likelihoods will be computed through addition in log_2 space instead of multiplication to
    # avoid underflow.
    # pylint: disable=assignment-from-no-return
    emission_matrix = np.log2(emission_matrix)
    transition_matrix = np.log2(transition_matrix)

    # Dynamic Programming
    # -------------------
    # The `paths` matrix will house the paths built by the Viterbi algorithm.
    # E.g. for trigram HMMs, `paths[k][u][v]` will contain the likelihood of the most likely path
    # suffixed with (u, v).
    #
    # Shape: (num_tokens + 1, [label_namespace_size] * (order - 1)).
    # The first dimension should be equal to the number of tokens in the input sequence + 1. This
    # allows the very first loop of the algorithm to reference the start to all paths, paths[0].
    # By beginning the loop at 1 and allowing the sequence to be 1-indexed, each path is properly
    # initialized with paths[0][START][STAART].
    #
    # The remaining (order - 1) dimensions, each of size `label_namespace_size` allow indexing
    # using (n - 1) grams of labels to find the most likely hidden label sequence at step k
    # suffixed with that(n - 1) gram.
    paths = np.zeros([num_tokens + 1] + [label_namespace_size] * (order - 1),)

    # The initial path starting at position 0 in the input sequence should ensure that only ngrams
    # prefixed with (n - 1) start tokens begin with probability 1 to prevent any paths not prefixed
    # with start tokens from being an optimal path. Note: log(1) = 0.
    paths.fill(float("-inf"))
    init_path_index = [0] + [start_token_id] * (order - 1)
    paths[tuple(init_path_index)] = 0

    # Backpointers
    # ------------
    # Backpointers at each step k of the algorithm record the index of the label w that maximizes
    # path[k][<(n - 1) gram prefix>] for some (n -1) gram prefix. Since we allow the sequence to be
    # 1-indexed to simplify the first iteration of the loop, as well as to better resemble the
    # algorithm's pseudocode, we must also 1-index the backpointer's first dimension.
    backpointers = np.zeros(
        [num_tokens + 1] + [label_namespace_size] * (order - 1), dtype=int
    )
    backpointers.fill(-1)  # A precaution to catch invalid paths.

    # The Viterbi Algorithm
    # ---------------------
    # Loop through the input sequence, exhaustively exploring all possible paths and storing
    # backpointers to all indices at each step that led to the maximal path suffixed with
    # that particular (n - 1) gram.
    #
    # E.g. For trigram HMMs, backpointers[k][u][v] contain the index to the label that led to the
    # highest scoring hidden sequence suffixed with u, v.
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

    # Final Transition
    # ----------------
    # `stop_transition` shape: label_namespace_size for (order - 1) dimensions.
    #
    # The Viterbi algorithm ends each path with a transition to the stop token.
    # Since transition_matrix[w][u][...][STOP] = q(STOP | w,u,...), we can collect all
    # q(STOP | w,u, ..) transitions (i.e. the submatrix such that indexing it with n - 1 tokens
    # [w, u, ...] results in q(STOP | w, u, ...)) by
    # 1. transposing (since transposition reverses the order of dimensions),
    # 2. indexing at end_token_id, and
    # 3. transposing the result.
    #
    # From here, we can add the stop transition in log-space to all paths appropriately by
    # broadcasting. Note that paths[num_tokens][...][w][u] reveals all paths suffixed by ...w,u
    # and `stop_transition` is the transition to STOP for all ngrams prefixed with ...w,u.
    stop_transition = (transition_matrix.T)[end_token_id].T
    r_final = paths[num_tokens] + stop_transition

    # Collect the log likelihood of the path chosen by Viterbi.
    log_likelihood = np.max(r_final)

    # Backtrace Algorithm
    # -------------------
    # To collect the final n - 1 labels in order to perform the backtrace,
    # we need to select the most likely path.
    # `paths[num_tokens] + stop_transition` is the cumulation of all paths,
    # the most likely path is then the path such that transition to stop
    # leads to the highest likelihood contained in this matrix. To begin
    # the backtrace, we need the label indices that index into this value.
    #
    # np.argmax() returns the index of the largest value in a multi-dim array,
    # but only *after* flattening. np.unravel_index() takes this index along with
    # the shape of the array to properly return the list of indices we need.
    output_indices = list(np.unravel_index(np.argmax(r_final), r_final.shape))

    if len(input_tokens) < (order - 1):
        # Special case: for input sequences of length less than the order minus 1.
        output_indices = output_indices[-len(input_tokens) :]
    else:
        for k in range(num_tokens - (order - 1), 0, -1):
            backpointer_index = [k + (order - 1)] + output_indices[: order - 1]
            yk = backpointers[tuple(backpointer_index)]
            output_indices.insert(0, yk)

    return {"log_likelihood": log_likelihood, "label_ids": output_indices}


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

    # Only the start tokens begin with a non -inf value.
    paths.fill(float("-inf"))
    init_path_index = [0] + [start_token_id] * (order - 1)
    paths[tuple(init_path_index)] = 0

    backpointers = np.zeros(
        [num_tokens + 1] + [label_namespace_size] * (order - 1), dtype=int
    )

    for k, token in enumerate(input_tokens, start=1):
        for u in range(label_namespace_size):
            for v in range(label_namespace_size):
                # For each ngram suffixed with (u, v), find a prefixing w that maximizes
                # the path suffixed with u, v:
                # paths[k][u][v] = max_{w in states} paths[k - 1][w][u] + q(v | w, u) + e(x_k | v)
                max_r, max_r_index = float("-inf"), -1
                for w in range(label_namespace_size):
                    r = (
                        paths[k - 1][w][u]
                        + transition_matrix[w][u][v]
                        + emission_matrix[token][v]
                    )

                    if r > max_r:
                        max_r = r
                        max_r_index = w

                # Update backpointers.
                backpointers[k][u][v] = max_r_index

                # Update paths.
                paths[k][u][v] = max_r

    # Backtrace
    # ---------
    # The index leading to the maximal path ending in STOP correspond to the final two tokens
    # in the maximal path.
    # In other words, since paths are constrained to end in STOP:
    #   max(u, v) paths[ number of tokens][u][v] + transition_matrix[u][v][end_token_id]
    # leads to the maximal penultimate and final tokens.
    penultimate_token, final_token = None, None
    max_final = float("-inf")
    for u in range(label_namespace_size):
        for v in range(label_namespace_size):
            r_final = paths[-1][u][v] + transition_matrix[u][v][end_token_id]
            if r_final > max_final:
                max_final = r_final
                penultimate_token = u
                final_token = v

    output_indices = [penultimate_token, final_token]
    for k in range(num_tokens - 2, 0, -1):
        yk = backpointers[k + 2][output_indices[0]][output_indices[1]]
        output_indices.insert(0, yk)

    # Collect the log likelihood of the path chosen by viterbi.
    log_likelihood = max_final

    return {"log_likelihood": log_likelihood, "label_ids": output_indices}
