def myMLP(lower_limit, upper_limit):
    import pandas as pd
    ds = pd.read_csv(r"Book1.csv")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = ds.iloc[lower_limit:upper_limit, :-1].values
    y = ds.iloc[lower_limit:upper_limit, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        classifier = MLPClassifier(hidden_layer_sizes=(3, 2), random_state=5, learning_rate_init=0.5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    return classifier, acc


def global_test(model):
    print("Building global model...")
    import pandas as pd
    ds = pd.read_csv(r"Book1.csv")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = ds.iloc[:, :-1].values
    y = ds.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(acc)

