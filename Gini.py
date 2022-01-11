def best_split(self) -> tuple:
    """
    Given the X features and Y targets calculates the best split
    for a decision tree
    """
    # Creating a dataset for spliting
    df = self.X.copy()
    df['Y'] = self.Y

    # Getting the GINI impurity for the base input
    GINI_base = self.get_GINI()

    # Finding which split yields the best GINI gain
    max_gain = 0

    # Default best feature and split
    best_feature = None
    best_value = None

    # Getting a random subsample of features
    n_ft = int(self.n_features * self.X_features_fraction)

    # Selecting random features without repetition
    features_subsample = random.sample(self.features, n_ft)

    for feature in features_subsample:
        # Droping missing values
        Xdf = df.dropna().sort_values(feature)

        # Sorting the values and getting the rolling average
        xmeans = self.ma(Xdf[feature].unique(), 2)

        for value in xmeans:
            # Spliting the dataset
            left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
            right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

            # Getting the Y distribution from the dicts
            y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0,
                                                                                                                  0), right_counts.get(
                1, 0)

            # Getting the left and right gini impurities
            gini_left = self.GINI_impurity(y0_left, y1_left)
            gini_right = self.GINI_impurity(y0_right, y1_right)

            # Getting the obs count from the left and the right data splits
            n_left = y0_left + y1_left
            n_right = y0_right + y1_right

            # Calculating the weights for each of the nodes
            w_left = n_left / (n_left + n_right)
            w_right = n_right / (n_left + n_right)

            # Calculating the weighted GINI impurity
            wGINI = w_left * gini_left + w_right * gini_right

            # Calculating the GINI gain
            GINIgain = GINI_base - wGINI

            # Checking if this is the best split so far
            if GINIgain > max_gain:
                best_feature = feature
                best_value = value

                # Setting the best gain to the current one
                max_gain = GINIgain

    return (best_feature, best_value)