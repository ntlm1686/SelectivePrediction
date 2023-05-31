import torch

class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.  
    """
    def __init__(self, features, dim_features:int, num_classes:int, init_weights=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
        """
        super(SelectiveNet, self).__init__()
        self.features = features
        self.dim_features = dim_features
        self.num_classes = num_classes
        
        # represented as f() in the original paper
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        # represented as g() in the original paper
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.aux_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        prediction_out = self.classifier(x)
        selection_out  = self.selector(x)
        auxiliary_out  = self.aux_classifier(x)

        return prediction_out, selection_out, auxiliary_out
