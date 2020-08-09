class Data:
    """
    Bundles train, test loaders with index to class mappings if required.
    """

    def __init__(self, train_loader, test_loader, classes=None):
        self.train = train_loader
        self.test = test_loader
        self.classes = classes
