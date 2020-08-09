import torch


class Loss_fn:

    def __init__(self, baseLoss, model, l1Factor=0.0, l2Factor=0.0):
        self.baseLoss = baseLoss
        self.l1Factor = l1Factor
        self.l2Factor = l2Factor
        self.model = model

    def __call__(self, predictions, targets):
        return self.baseLoss(predictions, targets) + self.l1Factor * compute_L1(
            self.model) + self.l2Factor * compute_L2(
            self.model)


def compute_L1(model):
    allWeights = torch.cat([x.view(-1) for x in model.parameters()])
    loss = torch.norm(allWeights, 1)
    return loss


def compute_L2(model):
    allWeights = torch.cat([x.view(-1) for x in model.parameters()])
    loss = torch.norm(allWeights, 2)
    return loss
