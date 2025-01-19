from torchmetrics.classification import MulticlassAUROC, MulticlassSpecificity, MulticlassRecall, MulticlassAveragePrecision


def eval_metrics(preds, target, num_classes, device):
    # initial the metrics
    specificity_metric = MulticlassSpecificity(num_classes=num_classes, average='macro').to(device)
    sensitivity_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    aucpr_metric = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    auroc_metric = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)

    # measure metrics
    specificity = specificity_metric(preds, target.argmax(dim=1))
    sensitivity = sensitivity_metric(preds, target.argmax(dim=1))
    bac = (specificity + sensitivity) / 2
    aucpr = aucpr_metric(preds, target.argmax(dim=1))
    auroc = auroc_metric(preds, target.argmax(dim=1))

    return bac, sensitivity, aucpr, auroc
