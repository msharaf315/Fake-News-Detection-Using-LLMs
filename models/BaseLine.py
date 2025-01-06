from torch import nn
import torch.nn.functional as F


# Create the baseline model
class BiLSTMTextClassifierModel(nn.Module):

    def __init__(self, vocab, embedding_dim, hidden_dim, number_of_labels):
        super(BiLSTMTextClassifierModel, self).__init__()
        self.number_of_labels = number_of_labels
        self.embedding = nn.Embedding(len(vocab), embedding_dim, vocab["<pad>"])
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        self.top_layer = nn.Linear(2 * hidden_dim, self.number_of_labels)
        self.relu = nn.ReLU()
        self.softmax = F.softmax

    def forward(self, x):
        embeddings = self.embedding(x)
        rnn_output, _ = self.rnn(embeddings)
        last_hidden = rnn_output[:, -1, :]
        top_layer_output = self.top_layer(self.relu(last_hidden))
        return self.softmax(self.relu(top_layer_output), dim=-1)
