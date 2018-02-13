
def expected_vs_actual(v_path, actual_path):
    # Calculates per word accuracy, returns correct # of labellings
    # and total # of labelings..

    tp = 0
    total = 0

    v_path = v_path.split()
    actual_path = actual_path.split()

    for i, vi in enumerate(v_path):
        ai = actual_path[i]
        if ai == vi:
            tp += 1

        total += 1

    return tp, total
