from used_packages import *


#%%

# Extract group numbers
def glasso_groups_func(X_original):

    group_sizes = np.array([len(np.unique(X_original[col])) for col in X_original])

    groups = np.concatenate(
        [size * [i+1] for i, size in enumerate(group_sizes)]
    ).flatten()

    original_labels = X_original.columns.tolist()

    return groups, original_labels

#%%

def glasso(X_train, y_train, X_test, y_test, X_original, c_start=-6, c_stop=2, c_num=10, alpha=0.01, l1_reg=0.05,
                     scoring='accuracy', no_groups=False, n_iter=100, tol=1e-5, cmap='Set1', title=[],
                     verbose=False, save_plot=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    screr = mean_absolute_percentage_error #get_scorer(scoring)
    count = 0
    group_lass_labels, labels = glasso_groups_func(X_original)
    if no_groups:
        groups = range(1, len(X_train.columns) + 1)
        group_plot = group_lass_labels
    else:
        groups = group_lass_labels
        group_plot = groups.copy()
    if verbose:
        print(f'===============\nOriginal Labels\n===============\n{labels}\n')
        print(f'===============\nGroup Numbers\n===============\n{groups}\n')
    coefs = []
    scores = []
    best_lambda = None
    best_score = np.inf
    np.random.seed(0)
    X_tr, X_ts, y_tr, y_ts = X_train_scaled, X_test_scaled, y_train, y_test #train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=0)

    count += 1
    group_lasso = GroupLasso(groups=groups, group_reg=alpha, l1_reg=l1_reg,
                             n_iter=n_iter,
                             tol=tol,
                             #scale_reg="inverse_group_size",
                             subsampling_scheme=1, warm_start=True)
    group_lasso.fit(X_tr, y_tr)

    mask = (np.abs(group_lasso.coef_.flatten()) > tol)

    if mask.sum() > 0:
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_tr[:, mask], y_tr)
        y_pred = ridge.predict(X_ts[:, mask])
    else:
        y_pred = np.full(y_ts.shape, y_tr.mean())

    coefs.append(group_lasso.coef_)
    scr = screr(y_ts, y_pred)
    if verbose:
        print(f'{count}/{c_num} lambda: {alpha:.2e} - Selected features: {mask.sum()} - {scoring} = {scr:.10}')
    scores.append(scr)
    if scr < best_score:
        best_score = scr
        best_lambda = alpha
        best_coefs = group_lasso.coef_.flatten()
        best_glass_model = group_lasso
        best_ypred = y_pred

    if best_lambda is None:
        raise ValueError("Best lambda not found.")

    return best_glass_model, best_coefs, best_lambda, best_ypred, np.array(groups), labels, best_score


#%%

def cluster_mean_shift(X_or, X_ohe, gr, groups_best, labels_best, coefs, labels_dict, band = 0.1, threshold=0.001, min_clusters=3, max_clusters=8, merge_middle_clusters=False, merge_threshold=0.025, save_plot=False, verbose=False, xticks=False, barwidth=0.5):

    # Get coefficients for the current group
    idx_group = np.argwhere(groups_best == gr)
    bar_series = pd.Series(coefs[idx_group].flatten(), index=X_or.columns[idx_group][:, 0]).sort_values(ascending=False)
    bars = bar_series.values
    bar_labels = bar_series.index


    or_labels = pd.Series(coefs[idx_group].flatten(), index=X_ohe.columns[idx_group][:, 0]).sort_values(ascending=False)
    bar_label_or = or_labels.index

    bars = np.sign(bars) * np.maximum(np.abs(bars) - threshold, 0)

    # Reshape for 1D clustering
    X = bars.reshape(-1, 1)

    # Extract label information to use later
    lab = []
    lab_or = []
    for l, l_or in zip(bar_labels, bar_label_or):
        reg = re.search(r'_(.*)', l)
        lab.append(reg.group(1))
        reg_or = re.search(r'_(.*)', l_or)
        lab_or.append(reg_or.group(1))
    main_label = labels_best[gr - 1]

    min_clusters = max(1, min_clusters)

    bandwidth = estimate_bandwidth(X, quantile=band)

    if bandwidth <= 0:
        data_range = np.max(X) - np.min(X)
        if data_range > 0:
            bandwidth = data_range / 20
        else:
            bandwidth = 0.1

    # Try different bandwidths to get the desired number of clusters
    attempts = 0
    max_attempts = 20

    while attempts < max_attempts:
        meanshift = MeanShift(bandwidth=bandwidth)
        clusters = meanshift.fit_predict(X)

        # Count number of clusters
        n_clusters = len(np.unique(clusters))
        title = f'{labels_best[gr - 1]} - {n_clusters} clusters (bandwidth={bandwidth:.4f}) - Mean Shift'

        # If we have too many clusters, increase bandwidth
        if n_clusters > max_clusters:
            bandwidth *= 1.5
        # If we have too few clusters, decrease bandwidth
        elif n_clusters < min_clusters:
            # Only try to create more clusters if there's enough variation
            if np.std(X) > 0:
                bandwidth *= 0.75
            else:
                break  # Can't create more clusters if data is too uniform
        else:
            break  # Number of clusters is acceptable

        attempts += 1

        # Safety check to prevent unreasonable bandwidths
        if bandwidth > np.max(X) - np.min(X) or bandwidth < 1e-10:
            break

    # Calculate mean absolute value per cluster
    cluster_means = {}
    for cluster_id in np.unique(clusters):
        # Get indices of features in this cluster
        cluster_mask = clusters == cluster_id
        # Calculate mean of absolute values
        cluster_means[cluster_id] = np.mean(np.abs(X[cluster_mask]))

    if merge_middle_clusters:

        # Identify clusters to merge (those with mean abs value below threshold)
        clusters_to_merge = [c for c, mean in cluster_means.items() if mean < merge_threshold]
        # Create new cluster assignments, merging low-impact clusters
        merged_clusters = clusters.copy()
        if clusters_to_merge:
            # Assign a new cluster ID for merged clusters (use max existing + 1)
            merge_id = max(np.unique(clusters)) + 1

            # Perform the merge
            for c in clusters_to_merge:
                merged_clusters[clusters == c] = merge_id
            clusters = merged_clusters

    # Find all unique labels without sorting them
    unique_labels = []
    for c in clusters:
        if c not in unique_labels:
            unique_labels.append(c)

    # Create mapping from original labels to sequential labels
    cluster_map = {old_label: new_label+1 for new_label, old_label in enumerate(unique_labels)}

    # Apply the mapping to get new cluster labels
    new_clusters = np.array([cluster_map[c] for c in clusters])

    lab_dict = dict(zip(lab, new_clusters))

    if verbose:
        print(lab)
        print(main_label)
        print(lab_dict)

    plt.figure(figsize=[12, 2])

    # Get the number of clusters for the colormap
    n_clusters = len(unique_labels)
    cmap = cm.get_cmap('RdBu', n_clusters)

    for i, (height, cluster) in enumerate(zip(bars, new_clusters)):
        plt.bar(i+1, height, color=cmap(cluster-1), width=barwidth)

    legend_elements = [plt.Rectangle((0,0), 1, 1, color=cmap(i))
                       for i in range(n_clusters)]
    '''plt.legend(legend_elements, [f'Cluster {i}' for i in range(1, n_clusters+1)],
               loc='best')'''
    if xticks:
        plt.xticks(ticks=range(1, len(X)+1), labels=lab_or, rotation=90)
    else:
        plt.xticks([])
    plt.yticks([])
    plt.title(f'{labels_dict[labels_best[gr - 1]]} - {n_clusters} clusters')
    plt.grid(False)
    if save_plot:
        plt.savefig(f'./clusters {n_clusters} - {labels_dict[labels_best[gr - 1]]}.pdf', bbox_inches='tight')
    plt.show()

    return lab_dict



#%%
models = []

def evaluate_models(models, X, y):
    assert(min(y) > 0)
    guessed_sales = np.array([model.guess(X) for model in models])
    mean_sales = guessed_sales.mean(axis=0)
    relative_err = np.absolute((y - mean_sales) / y)
    result = np.sum(relative_err) / len(y)
    return result

#%%
def sample(X, y, n, seed=0):
    '''random samples'''
    num_row = X.shape[0]
    np.random.seed(seed)
    indices = np.random.randint(num_row, size=n)
    return X[indices, :], y[indices], indices, seed

