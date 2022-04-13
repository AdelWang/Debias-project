# classifier for predicting the overlap of the sentence pair
class classifier_pred(nn.Module):
    def __init__(self, config, scale = 2, activation=nn.Sigmoid()):
        super(classifier_pred, self).__init__()
        self.config = config
        self.fc_pred_hidden_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size * scale)
        self.fc_pred_hidden_2 = nn.Linear(self.config.hidden_size * scale, self.config.hidden_size)
        self.fc_pred = nn.Linear(self.config.hidden_size, 1)
        self.activation_hidden = nn.Tanh()
        self.activation = activation
        self.dropouts = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_hidden_states, y=None, loss_fn=nn.MSELoss(reduction='mean')):
        if y is not None:
            y = y.reshape(-1,1)
        last_hidden = input_hidden_states[-1]
        feature = torch.mean(last_hidden, axis = 1)
        h_hidden = self.fc_pred_hidden_1(self.dropouts(feature))
        pred_hidden = self.activation_hidden(h_hidden)
        h_hidden = self.fc_pred_hidden_2(self.dropouts(pred_hidden))
        pred_hidden = self.activation_hidden(h_hidden)
        h = self.fc_pred(self.dropouts(pred_hidden))
        predict = self.activation(h)
        if loss_fn is not None:
            loss = loss_fn(predict.float(), y.float())
            return predict , loss 
        return predict
