import torch
from torch import nn

class CNNLSTM(nn.Module):

    def __init__(self, dropout=0.2, kernel=4, filters=200, td_layer="BILSTM", num_classes=15,
                 length_ts=64, c_in=13):
        super(CNNLSTM, self).__init__()
        self.td_layer = td_layer
        print('Build IMU with ' + self.td_layer + '...')
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=filters, kernel_size=kernel, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(filters),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(filters),
            nn.Dropout(dropout))


        if self.td_layer == "LSTM":
            self.model = nn.LSTM(input_size=filters, hidden_size=100, bidirectional=False)
            self.model_act = nn.ReLU6()
            out_features_model = int(100 * ((length_ts / 2) / 2))
            # td = layers.LSTM(units=100, activation=tf.nn.relu6, return_sequences=True)(drop2)
        elif self.td_layer == "BILSTM":
            # td = layers.Bidirectional(layers.LSTM(units=60, activation="relu", return_sequences=True))(drop2)
            self.model = nn.LSTM(input_size=filters, hidden_size=60, bidirectional=True)
            self.model_act = nn.Tanh()
            out_features_model = int(120 * ((length_ts / 2) / 2))
        else:
            raise ValueError("Not implemented Layer: " + str(td_layer))
        
        self.classifier = nn.Sequential(
           nn.Linear(in_features=out_features_model, out_features=100),
           nn.BatchNorm1d(100),
           nn.Linear(in_features=100, out_features=200),
           nn.Linear(in_features=200, out_features=num_classes)
        )

    def forward(self, x, probits=False):
        x = self.feature_extraction(x)
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print (name, param.data)
        # print("nonzero inputs: ", torch.sum(x!= 0))
        if (self.td_layer == "LSTM") or (self.td_layer == "BILSTM"):
            x = self.model_act(self.model(x.permute(0,2,1))[0])
            x = x.flatten(1)
        else:
            selfmodelx = self.model(x)
            # nonzeroweights = torch.sum(selfmodelx != 0)
            # print("nonzero weights after tempconvnet: ", nonzeroweights)
            x = self.model_act(selfmodelx)
            # print("nonzero weights lost after relu(tempconvnet): ", nonzeroweights - torch.sum(x != 0))
            x = x.flatten(1)
            
        logits = self.classifier(x)
        
        if not probits:
            return logits
        else:
            return torch.nn.functional.softmax(logits, dim=1)

if __name__ == "__main__":
    model = CNNLSTM(0.2, 4, 200, td_layer="BILSTM", num_classes=60, c_in=13)
    print(model)