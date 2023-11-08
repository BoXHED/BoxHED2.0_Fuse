from torch import nn

class ClinicalLSTM(nn.Sequential):
    '''
    encoder: LSTM 
    decoder: Linear (hidden_size -> 32 -> 1)
    '''
    
    def __init__(self):
            super(ClinicalLSTM, self).__init__()
            
            self.intermediate_size = 1536
            self.num_attention_heads = 12
            self.attention_probs_dropout_prob = 0.1
            self.hidden_dropout_prob = 0.1
            self.hidden_size_encoder = 64
            self.n_layers = 2
            self.hidden_size_xlnet = 64
            
            self.encoder = nn.LSTM(input_size = self.hidden_size_encoder, hidden_size = self.hidden_size_encoder, num_layers = 2, bidirectional = True, batch_first = False)    
            self.decoder = nn.Sequential(
                nn.Dropout(p=self.hidden_dropout_prob),
                nn.Linear(self.hidden_size_encoder*2, 32),
                nn.ReLU(True),
                #output layer
                nn.Linear(32, 1)
            )
            
    def forward(self, xlnet_outputs):
           
            self.encoder.flatten_parameters()
            output, (a, b) = self.encoder(xlnet_outputs)

            print('a.shape', a.shape)
            print('b.shape', b.shape)
    
            last_layer = output[-1]
            print('last_layer shape:', last_layer.shape)
            score = self.decoder(last_layer)
            
            return score