import numpy as np

def _assign(costs):
    valid = costs < 1
    single = valid & (valid.sum(axis=1)[:, None] == 1) & (valid.sum(axis=0)[None, :] == 1)
    rr, cc = np.where(single)
    result = list(zip(map(int, rr), map(int, cc)))
    valid[rr, :] = False
    remaining = set(np.flatnonzero(valid.any(axis=1)))
    while remaining:
        rows = {min(remaining)}
        columns = set()
        while True:
            next_columns = set(np.flatnonzero(valid[sorted(rows)].any(axis=0)))
            next_rows = set(np.flatnonzero(valid[:, sorted(next_columns)].any(axis=1)))
            if next_rows == rows and next_columns == columns:
                break
            rows, columns = (next_rows, next_columns)
        remaining -= rows
        rows, columns = (sorted(rows), sorted(columns))
        group = costs[np.ix_(rows, columns)]
        if min(group.shape) == 1:
            r, c = np.unravel_index(group.argmin(), group.shape)
            result.append((rows[r], columns[c]))
        elif max(group.shape) <= 24:
            result.extend(((rows[r], columns[c]) for r, c in _hungarian(group)))
        else:
            rr, cc = np.where(group < 1)
            used_rows, used_columns = (set(), set())
            for index in np.argsort(group[rr, cc], kind='stable'):
                r, c = (int(rr[index]), int(cc[index]))
                if r not in used_rows and c not in used_columns:
                    result.append((rows[r], columns[c]))
                    used_rows.add(r)
                    used_columns.add(c)
    return result

def _hungarian(costs):
    n, m = costs.shape
    if not n or not m:
        return []
    a = np.concatenate((costs, np.ones((n, n))), axis=1)
    columns = a.shape[1]
    u, v = (np.zeros(n + 1), np.zeros(columns + 1))
    p, way = (np.zeros(columns + 1, dtype=int), np.zeros(columns + 1, dtype=int))
    for i in range(1, n + 1):
        p[0], j0 = (i, 0)
        minimum, used = (np.full(columns + 1, np.inf), np.zeros(columns + 1, dtype=bool))
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = (np.inf, 0)
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = a[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], way[j] = (current, j0)
                if minimum[j] < delta:
                    delta, j1 = (minimum[j], j)
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] and costs[p[j] - 1, j - 1] < 1]
