def take_subset(X, y, start_fraction=None, end_fraction=None, start=None, stop=None):
    if start_fraction is not None:
        n = X.shape[0]
        subset_end = int(start_fraction * n)
        X = X[0:subset_end, :]
        if y.ndim == 2:
            y = y[0:subset_end, :]
        elif y.ndim == 1:
            y = y[0:subset_end]
        else:
            raise ValueError("not able to handle a y with > 2 dims")
    elif end_fraction is not None:
        n = X.shape[0]
        subset_start = int((1 - end_fraction) * n)
        X = X[subset_start:, :]
        if y.ndim == 2:
            y = y[subset_start:, :]
        elif y.ndim == 1:
            y = y[subset_start:]
        else:
            raise ValueError("not able to handle a y with > 2 dims")
    elif start is not None:
        X = X[start:stop, :]
        if y is not None:
            if y.ndim == 2:
                y = y[start:stop, :]
            elif y.ndim == 1:
                y = y[start:stop]
            else:
                raise ValueEerror("not able to handle a y with > 2 dims")

    return X, y
