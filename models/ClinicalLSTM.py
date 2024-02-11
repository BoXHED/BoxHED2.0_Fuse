from torch import nn

class ClinicalLSTM(nn.Sequential):
    '''
    encoder: LSTM 
    decoder: Linear (hidden_size -> 32 -> 1)
    '''
    
    def __init__(self, bidirectional = False):
            super(ClinicalLSTM, self).__init__()
            
            self.intermediate_size = 1536
            self.num_attention_heads = 12
            self.attention_probs_dropout_prob = 0.1
            self.hidden_dropout_prob = 0.1
            self.hidden_size_encoder = 64
            self.n_layers = 2
            self.hidden_size_xlnet = 64
            
            self.encoder = nn.LSTM(input_size = self.hidden_size_encoder, hidden_size = self.hidden_size_encoder, num_layers = 2, bidirectional = bidirectional, batch_first = False)    
            decoder_size = 2 * self.hidden_size_encoder if bidirectional else self.hidden_size_encoder
            self.decoder = nn.Sequential(
                nn.Dropout(p=self.hidden_dropout_prob),
                nn.Linear(decoder_size, 32),
                nn.ReLU(True),
                #output layer
                nn.Linear(32, 1),
            )
            
    def forward(self, xlnet_outputs):
        '''
        assuming max_num_notes = 32, doc_emb_size = 64
        encode a (32 x batch_size x 64) tensor into a (32, batch_size, hidden_size * num_directions) last hidden state tensor   
        decode the last hidden layer (batch_size, hidden_size * num_directions) into a (batch_size x 1) tensor
        '''
           
        self.encoder.flatten_parameters()
        output, (a, b) = self.encoder(xlnet_outputs)


        last_layer = output[-1]
        score = self.decoder(last_layer)

        # print('ENCODER IN:', xlnet_outputs.shape)
        # print('ENCODER OUT:', output.shape)
        # print('DECODER IN:', last_layer.shape)
        # print('DECODER OUT:', score.shape)
        
        return score