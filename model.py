from sklearn.neural_network import MLPClassifier
import numpy as np


class Model(MLPClassifier):
    def __init__(self, zipped_averaged_weights, zipped_averaged_biases):
        super().__init__()
        self.zipped_averaged_weights = zipped_averaged_weights
        self.zipped_averaged_biases = zipped_averaged_biases

    def _init_coef(self, fan_in, fan_out, dtype):
        if self.activation == 'logistic':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation in ('identity', 'tanh', 'relu'):
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        else:
            raise ValueError("Unknown activation function %s" %
                             self.activation)
        coef_init = self.zipped_averaged_weights.astype(dtype, copy=False)
        intercept_init = self.zipped_averaged_biases.astype(dtype, copy=False)
        return coef_init, intercept_init

    def myMLP(self, params):
        round_no, client_no, lower_limit, upper_limit, = params[0], params[1], params[2], params[3]
        # print("\n", round_no, client_no, layers, nodes, lower_limit, upper_limit,"\n")
        import pandas as pd
        ds = pd.read_csv(r"C:\Users\ADMIN\pythonFLProject\Book2.csv")

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

            classifier = MLPClassifier(hidden_layer_sizes=(14, 8), random_state=5, solver="sgd",
                                       learning_rate="constant", learning_rate_init=0.0000001)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve
            acc = accuracy_score(y_test, y_pred) * 100
            loss = classifier.loss_
            f1sr = f1_score(y_test, y_pred, average="macro") * 100
            recall = recall_score(y_test, y_pred, average="macro") * 100

            print("R.No ", round_no, ":", sep="", end=" ")
            print("acc {:.3f}".format(acc), end="   ")
            print("loss {:.3f}".format(loss), end="   ")
            print("f1-score {:.3f}".format(f1sr), end="   ")
            print("recall {:.3f}".format(recall), end="   ")
            metrics = [acc, loss, f1sr, recall]

            # store roc curve image in the folder
            import os
            previous_dir = os.getcwd()
            os.chdir(r"C:\Users\ADMIN\pythonFLProject\outputs")
            filename = str(round_no) + "_" + str(client_no) + "_acc-{:.2f}".format(acc) + ".png"
            import matplotlib.pyplot as plt
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve for Round " + str(round_no) + " Client " + str(client_no))
            plt.savefig(filename)
            plt.close()
            # go to previous directory
            os.chdir(previous_dir)
        return classifier.coefs_, classifier.intercepts_, metrics
