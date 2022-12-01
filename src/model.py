from sklearn.neural_network import MLPClassifier
import numpy as np


class Model(MLPClassifier):
    def __init__(self, zipped_averaged_weights):
        self.zipped_averaged_weights = zipped_averaged_weights

    def _init_coef(self, fan_in, fan_out):
        if self.activation == 'logistic':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            raise ValueError("Unknown activation function %s" %
                             self.activation)
        coef_init = self.zipped_averaged_weights
        intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)

        return coef_init, intercept_init

    def myMLP(self, lower_limit=0, upper_limit=178000):
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

        import warnings
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            classifier = MLPClassifier(hidden_layer_sizes=(3, 2), random_state=5, solver="sgd", learning_rate="invscaling", learning_rate_init=0.8)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)
            print(acc)
        return classifier.coefs_

    def local_test(lower_limit, upper_limit, classifier):
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

        from sklearn.metrics import accuracy_score
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
