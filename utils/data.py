from torch.utils.data import Subset

def subset_first_n_groups(dataset, n: int) -> Subset:
    if dataset.groups is None:
        raise ValueError("Dataset does not expose groups")

    if n < 1:
        raise ValueError("n must be >= 1")

    seen = []
    selected_indices = []

    for idx, group in enumerate(dataset.groups):
        if group not in seen:
            if len(seen) >= n:
                break
            seen.append(group)

        if group in seen:
            selected_indices.append(idx)

    subset = Subset(dataset, selected_indices)
    subset.samples = [dataset.samples[i] for i in selected_indices]
    subset.groups = [dataset.groups[i] for i in selected_indices]
    return subset
