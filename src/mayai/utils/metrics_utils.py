def compute_accuracy(prediction, target):
    return 100 * prediction.eq(target.view_as(prediction)).sum().item() / float(len(prediction))
