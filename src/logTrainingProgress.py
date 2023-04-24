import numpy as np

def logTrainingProgress(tensorboard, idx_epoch, mode, loss, accuracy, class_true, class_pred):
    assert len(class_true) == len(class_pred)
    true_positive_rate  = np.logical_and(class_true == 0, class_pred == 0).sum() / float(len(class_true))
    false_positive_rate = np.logical_and(class_true == 0, class_pred == 1).sum() / float(len(class_true))
    true_negative_rate  = np.logical_and(class_true == 1, class_pred == 1).sum() / float(len(class_true))
    false_negative_rate = np.logical_and(class_true == 1, class_pred == 0).sum() / float(len(class_true))

    precision = true_positive_rate / (true_positive_rate + false_positive_rate)
    recall = true_positive_rate / (true_positive_rate + false_negative_rate)

    print("%s: Avg loss = %1.6f, accuracy = %1.2f%%" % (mode.capitalize(), loss, 100 * accuracy))
    print(" rates: TP = %1.2f%%, FP = %1.2f%%, TN = %1.2f%%, FN = %1.2f%% (precision = %1.2f%%, recall = %1.2f%%)" % \
      (100 * true_positive_rate, 100 * false_positive_rate, 100 * true_negative_rate, 100 * false_negative_rate,
       100 * precision, 100 * recall))

    tensorboard.add_scalar("Loss/%s" % mode, loss, global_step=idx_epoch)
    tensorboard.add_scalar("Accuracy/%s" % mode, 100 * accuracy, global_step=idx_epoch)
    tensorboard.add_pr_curve("ROC_curve/%s" % mode, np.array(class_true), np.array(class_pred), global_step=idx_epoch)
    tensorboard.add_scalar("false_positives/%s" % mode, false_positive_rate, global_step=idx_epoch)
    tensorboard.add_scalar("false_negatives/%s" % mode, false_negative_rate, global_step=idx_epoch)
    tensorboard.add_scalar("precision/%s" % mode, precision, global_step=idx_epoch)
    tensorboard.add_scalar("recall/%s" % mode, recall, global_step=idx_epoch)
    tensorboard.add_histogram("tauClassifier_sig/%s" % mode, class_pred[class_true == 1], global_step=idx_epoch)
    tensorboard.add_histogram("tauClassifier_bgr/%s" % mode, class_pred[class_true == 0], global_step=idx_epoch)
