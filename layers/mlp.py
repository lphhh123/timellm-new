import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim=1024,layer_num = 4, dropout=0.5, device=None):
        super(MLP, self).__init__()
        self.input_dim = input_shape[0] * input_shape[1]
        self.output_dim = output_shape[0] * output_shape[1]
        self.output_shape = output_shape
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建多层 MLP 
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
        for _ in range(layer_num - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # 使用 nn.Sequential 构建网络
        self.classifier = nn.Sequential(*layers)
        self.classifier.to(self.device)

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.hidden_dim, self.output_dim)
        # )
        # self.classifier.to(self.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x.view(x.size(0), self.output_shape[0], self.output_shape[1])
    
    
        


def main():
    input_shape = (8, 6)
    output_shape = (8, 1)
    classifier = MLP(output_shape, input_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = torch.randn(128, 8, 1).to(device)

    outputs = classifier(features)

    print("outputs shape:", outputs.shape)


if __name__ == '__main__':
    main()
