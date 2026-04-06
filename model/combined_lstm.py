
import torch
import torch.nn as nn

class CombinedLSTMModel(nn.Module):
    def __init__(
        self,
        token_vocab_size,  #characteristic of the tokenizer
        token_embedding_dim, #number of token per instruction
        access_feature_size, #number of trace features
        hidden_dim, #matrix size of then hidden dimension of the LSTM layer
        output_dim, #number of output of our model
        num_layers, #number of consecutive LSTM layers
        dropout, #dropout rate between the LSTM layers
    ):
        super(CombinedLSTMModel, self).__init__()
        self.token_embedding = nn.Embedding(token_vocab_size, token_embedding_dim)
        # input size must be the sum of flattened token embedding size and access feature size
        self.lstm = nn.LSTM(
            input_size=token_embedding_dim**2 + access_feature_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, token_features, access_features):
        # emded the token features
        embedded_tokens = self.token_embedding(token_features)
        # assume access_features is already of the appropriate size and requires no embedding
        # LSTM can handle variable-length sequences, if necessary, but here we focus on batch size flexibility
        elongated_embeddings = embedded_tokens.view(token_features.size(0), token_features.size(1), -1)
        combined_features = torch.cat(
            (elongated_embeddings, access_features), dim=2
            # concatenate along the feature dimension
        )
        lstm_out, _ = self.lstm(combined_features)
        # using the last time step's output
        output = self.fc(lstm_out[:, -1, :])
        return output
        