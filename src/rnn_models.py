import torch.nn as nn


class SimpleMidiRNN(nn.Module):
  def __init__(self,  input_size, hidden_size, num_layers, bidirectional, rnn_class = nn.GRU, notes_num = None, nan_class = False):
    super(SimpleMidiRNN, self).__init__()
    if not notes_num:
      notes_num = input_size
    out_size = notes_num
    if nan_class:
      out_size += 1
    self.rnn = rnn_class(input_size = input_size, hidden_size=hidden_size, num_layers = num_layers, dropout = 0.1, bidirectional = bidirectional, batch_first = True)
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(),
        nn.BatchNorm1d(seq_length * 2 if bidirectional else seq_length),
        nn.Linear(seq_length * 2 if bidirectional else seq_length, out_size)
    )
  

  def forward(self, x):
    x = torch.squeeze(x, dim = 1)
    x = torch.transpose(x, 1, 2)
    # x = (batch, midi_length, notes_num)
    output, _ = self.rnn(x)
    # output = (batch, midi_length, hidden_size (* 2 if bidirectional))
    return  self.fc(output[:, -1, :]) # Take RNN output only from the last position    



class ConvMidiRNN(nn.Module):
  def __init__(self, notes_num, nan_class = False):
    super(ConvMidiRNN, self).__init__()
    self.conv = nn.Sequential(
        # (bs, 1, notes_num, midi_length)
        nn.Conv2d(1, 32, 3, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        # (bs, 32, notes_num, midi_length)
        nn.Conv2d(32, 128, (notes_num, 1)),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
    )
    # 128  is a number of features in the hidden state
    self.rnn = SimpleMidiRNN(128, seq_length, 2, True, notes_num = notes_num, nan_class = nan_class)


  def forward(self, x):
    x = self.conv(x)
    x = torch.transpose(x, 1, 2)
    return self.rnn(x)
