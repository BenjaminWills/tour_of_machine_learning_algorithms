from collections import defaultdict
import numpy as np


class evaluate:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        # Save data and predictions
        self.y_true = y_true
        self.y_pred = y_pred

        # Define unique classes
        self.unique_classes = np.unique(y_true)
        self.num_unique_classes = len(self.unique_classes)

        # Calculate confusion matrix
        self.confusion_matrix = self.confusion_matrix()

        # Calculate scores of interest
        self.scores = {}

        # Calculate true positives, false positives and false negatives for each class
        self.tp_fp_fn = self.calculate_tp_fp_fn()

        # Calculate precision and recall for each class
        self.precision_recall = self.calculate_precision_and_recall()

        # Calculate F1 score for each class
        self.f1_score = self.calculate_f1_score()

        # Calculate accruacy
        self.scores["accuracy"] = self.accuracy()
        self.scores["tp_fp_fn"] = self.tp_fp_fn
        self.scores["precision_recall"] = self.precision_recall
        self.scores["f1_score"] = self.f1_score

    def confusion_matrix(self) -> np.ndarray:
        """Calculate the confusion matrix of the model.

        Parameters
        ----------
        y_true : np.ndarray
            The true values of the target variable.
        y_pred : np.ndarray
            The predicted values of the target variable.

        Returns
        -------
        np.ndarray
            The confusion matrix of the model.
        """

        # The multi-class confusion matrix can be defined as follows:
        confusion_matrix = np.zeros((self.num_unique_classes, self.num_unique_classes))

        for class_1 in range(self.num_unique_classes):
            for class_2 in range(self.num_unique_classes):
                pred_class_1_and_true_class_2 = np.sum(
                    (self.y_true == self.unique_classes[class_1])
                    & (self.y_pred == self.unique_classes[class_2])
                )
                confusion_matrix[class_1, class_2] = pred_class_1_and_true_class_2

        return confusion_matrix

    def accuracy(self) -> float:
        """Calculate the accuracy of the model.

        Parameters
        ----------
        y_true : np.ndarray
            The true values of the target variable.
        y_pred : np.ndarray
            The predicted values of the target variable.

        Returns
        -------
        Float
            The accuracy of the model.
        """
        return self.confusion_matrix.trace() / len(self.y_true)

    def calculate_tp_fp_fn(self) -> None:
        """Calculate the true positives, false positives and false negatives for each class."""
        tp_fp_fn_storage = defaultdict(dict)
        for class_ in range(self.num_unique_classes):
            true_positive = self.confusion_matrix[class_, class_]
            false_positive = np.sum(self.confusion_matrix[:, class_]) - true_positive
            false_negative = np.sum(self.confusion_matrix[class_, :]) - true_positive

            tp_fp_fn_storage[class_] = {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        return dict(tp_fp_fn_storage)

    def calculate_precision_and_recall(self) -> float:
        """Calculate the precision and recall of the model.

        Parameters
        ----------
        y_true : np.ndarray
            The true values of the target variable.
        y_pred : np.ndarray
            The predicted values of the target variable.

        Returns
        -------
        Float
            The precision and recall of the model.
        """

        precision_recall_storage = defaultdict(dict)
        for class_ in range(self.num_unique_classes):
            true_positive = self.tp_fp_fn[class_]["true_positive"]
            false_positive = self.tp_fp_fn[class_]["false_positive"]
            false_negative = self.tp_fp_fn[class_]["false_negative"]

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            precision_recall_storage[class_] = {
                "precision": precision,
                "recall": recall,
            }
        return dict(precision_recall_storage)

    def calculate_f1_score(self) -> float:
        """Calculate the F1 score of the model.

        Parameters
        ----------
        y_true : np.ndarray
            The true values of the target variable.
        y_pred : np.ndarray
            The predicted values of the target variable.

        Returns
        -------
        Float
            The F1 score of the model.
        """
        f1_score_storage = defaultdict(dict)
        for class_ in range(self.num_unique_classes):
            precision = self.precision_recall[class_]["precision"]
            recall = self.precision_recall[class_]["recall"]

            f1_score = 2 * (precision * recall) / (precision + recall)

            f1_score_storage[class_] = f1_score
        return dict(f1_score_storage)
