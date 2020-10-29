import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from torch.autograd import Variable


class ImageEncoder(nn.Module):
    '''Outputs embedding given image input'''
    def __init__(self, output_dim):
        nn.Module.__init__(self)
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.model(x)
        return x


class TextEncoder(nn.Module):
    '''Outputs embedding given text input'''
    def __init__(self, output_dim, input_size=768):
        nn.Module.__init__(self)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=256,
                            num_layers=1, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=output_dim)
    
    def forward(self, utterances, hidden_init=None):
        '''
        Args:
            utterances: (batch_size, n_frames, 768) 
            hidden_init: initial hidden state of the LSTM as a tensor of 
                shape (num_layers, batch_size, hidden_size). 
                Will default to a tensor of zeros if None.
        '''
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        summary, _ = torch.max(out, dim=1)
        output = self.linear(summary)
        return output


class SpeechEncoder(nn.Module):
    def __init__(self, num_classes, input_size=40):
        nn.Module.__init__(self)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=256,
                            num_layers=2, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=num_classes)# 256)
    
    def forward(self, utterances, hidden_init=None):
        '''
        Args:
            utterances: batch of mel-scale filterbanks of same duration 
                as a tensor of shape 
                    (batch_size, n_frames, n_channels) 
            hidden_init: initial hidden state of the LSTM as a tensor of 
                shape (num_layers, batch_size, hidden_size). 
                Will default to a tensor of zeros if None.
        '''
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        summary, _ = torch.max(out, dim=1)
        output = self.linear(summary)
        return output


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, vocab_size, emb_mat, bidirectional=True):
        '''
        Args:
            vocab_size: note might not equal emb_vocab_size
                always >= emb_vocab_size though
                includes pad token and unknown word token
            emb_mat: numpy array with shape (emb_vocab_size, emb_dim)
        '''
        super(SimpleLSTM, self).__init__()
        emb_dim = emb_mat.shape[1]
        self.emb_layer = nn.Embedding(vocab_size, emb_dim)
        num_new_vocab = vocab_size - emb_mat.shape[0]
        extra_embs = np.random.normal(0.0, 1.0, size=(num_new_vocab, emb_dim))
        new_emb_mat = np.concatenate([emb_mat, extra_embs], 0)
        self.emb_layer.weight.data.copy_(torch.from_numpy(new_emb_mat))
        self.lstm = nn.LSTM(emb_dim, hidden_dim, 1, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, seq_len)
        '''
        x = self.emb_layer(x)
        out, (hidden, cell) = self.lstm(x, None)
        summary, _ = torch.max(out, dim=1)
        output = self.fc(summary)
        return output


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, x, h=None, c=None):
        '''
        Args:
            x: (batch size, input_dim)
            h: (n layers, batch size, hid dim)
            c: (n layers, batch size, hid dim)

        Return:
            pred: (batch size, output dim)
        '''
        x = x.unsqueeze(0)
        output, (h, c) = self.rnn(x, (h, c))
        pred = self.fc_out(output.squeeze(0))
        return pred, h, c


class ReptileModel(nn.Module):

    def __init__(self, device):
        nn.Module.__init__(self)
        self.device = device

    def point_grad_to(self, target):
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = Variable(torch.zeros(p.size())).cuda(self.device)
                else:
                    p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.zero_()
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda


class AEMetaModel(ReptileModel):
    def __init__(self, fc_dim, num_classes, device):
        ReptileModel.__init__(self, device)
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.encoder = SpeechEncoder(fc_dim)
        self.num_feats = 40
        self.decoder = LSTMDecoder(self.num_feats, self.num_feats, fc_dim, 1)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc_dim, num_classes)
        )
    
    def forward_ae(self, x, teacher_forcing_ratio=0.5):
        '''
        Args:
            x: (batch_size, seq_len, num_feats)
            teacher_forcing_ratio: probability to use teacher forcing
                e.g. if teacher_forcing_ratio is 0.75 we use ground-truth 
                inputs 75% of the time
        
        Return:
            outputs: shape (batch_size, seq_len, num_feats)
        '''
        out = self.encoder(x)
        hidden = out.unsqueeze(0).cuda(self.device)
        cell = torch.zeros_like(hidden).cuda(self.device)
        curr_input = torch.zeros(out.shape[0], self.num_feats).cuda(self.device)
        outputs = []
        seq_len = x.shape[1]
        for t in range(seq_len):
            output, hidden, cell = self.decoder(curr_input, hidden, cell)
            outputs.append(output)
            teacher_force = random.random() < teacher_forcing_ratio
            curr_input = x[:,t,:] if teacher_force else output
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2)
        return outputs

    def forward_audio(self, x):
        out = self.encoder(x)
        out = self.classifier(out)
        return out

    def forward(self, x):
        out = self.encoder(x)
        out = self.classifier(out)
        return out

    def clone(self):
        clone = AEMetaModel(self.fc_dim, self.num_classes, self.device)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda(self.device)
        return clone


class MergedMetaModel(ReptileModel):
    def __init__(self, fc_dim, num_classes, vocab_size, emb_mat, device):
        ReptileModel.__init__(self, device)
        
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_mat = emb_mat

        self.audio_fc = nn.Linear(40, 64)
        self.text_fc = nn.Linear(256, 64)

        emb_dim = emb_mat.shape[1]
        self.emb_layer = nn.Embedding(vocab_size, emb_dim)
        num_new_vocab = vocab_size - emb_mat.shape[0]
        extra_embs = np.random.normal(0.0, 1.0, size=(num_new_vocab, emb_dim))
        new_emb_mat = np.concatenate([emb_mat, extra_embs], 0)
        self.emb_layer.weight.data.copy_(torch.from_numpy(new_emb_mat))

        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=256,
                            num_layers=2, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=256, out_features=fc_dim)

        self.clf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc_dim, num_classes)
        )

    def forward_audio(self, x_audio):
        '''
        Args:
            x_audio: (batch_size, n_frames, n_channels)
        '''
        x_audio = self.audio_fc(x_audio)
        out, (hidden, cell) = self.lstm(x_audio)
        summary, _ = torch.max(out, dim=1)
        output = self.linear(summary)
        out = self.clf(output)
        return out

    def forward_text(self, x_text):
        x = self.emb_layer(x_text)
        x = self.text_fc(x)
        out, (hidden, cell) = self.lstm(x)
        summary, _ = torch.max(out, dim=1)
        output = self.linear(summary)
        out = self.clf(output)
        return out
    
    def forward(self, x):
        return self.forward_audio(x)    

    def clone(self):
        clone = MergedMetaModel(self.fc_dim, self.num_classes, self.vocab_size, self.emb_mat, self.device)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda(self.device)
        return clone


class MetaModel(ReptileModel):
    '''enc1 is the modality without clf labels'''
    def __init__(self, enc1, enc2, fc_dim, num_classes, device):
        ReptileModel.__init__(self, device)
        self.enc1 = enc1
        self.enc2 = enc2
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.align_layer = nn.Sequential(nn.Linear(fc_dim, fc_dim))
        self.clf = nn.Sequential(nn.ReLU(), nn.Linear(fc_dim, num_classes))

    def forward_align(self, x1, x2):
        out1 = self.enc1(x1)
        out1 = self.align_layer(out1)
        out2 = self.enc2(x2)
        return out1, out2

    def forward1(self, x1):
        encode1 = self.enc1(x1)
        align1 = self.align_layer(encode1)
        out = self.clf(align1)
        return out

    def forward2(self, x2):
        out = self.enc2(x2)
        out = self.clf(out)
        return out
    
    def forward(self, x):
        return self.forward1(x)

    def clone(self):
        clone = MetaModel(self.enc1, self.enc2, self.fc_dim, self.num_classes, self.device)
        clone.load_state_dict(self.state_dict())
        if self.is_cuda():
            clone.cuda(self.device)
        return clone
