import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class Conv1DLinear(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 kernel_size=2,
                 pool_size=2
                 ):
        super(Conv1DLinear, self).__init__()
        self.preprocess = Preprocessor()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        # size of output
        lout = 3750 - kernel_size + 1
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = math.floor(lout / pool_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = lout - kernel_size + 1
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = math.floor(lout / pool_size)
        print('lout: ', lout)
        if isinstance(output_size, (list, tuple)):
            self.out = nn.ModuleList(
                [nn.Linear(hidden_size * lout, o) for o in output_size]
            )
        else:
            self.out = nn.Linear(hidden_size * lout, output_size)
        self.nl = nn.ReLU()

    def forward(self, x, noise=None):
        x = self.preprocess(x)
        if noise is not None:
            x = x + noise
        x = self.nl(self.pool1(self.conv1(x)))
        x = self.nl(self.pool2(self.conv2(x)))
        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(x.view(-1, x.size(1) * x.size(2))
                  ) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(x.view(-1, x.size(1) * x.size(2)))
        return pred


class Conv1DBNLinear(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 kernel_size=2,
                 pool_size=2,
                 dropout=0
                 ):
        super(Conv1DBNLinear, self).__init__()
        self.preprocess = Preprocessor()
        self.batch_norm0 = nn.BatchNorm1d(input_size)

        lout = 3750

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout5 = nn.Dropout(p=0.5)

        input_size = hidden_size
        hidden_size = hidden_size // 2

        self.conv3 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        input_size = hidden_size

        self.conv5 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        if isinstance(output_size, (list, tuple)):
            self.out = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size * lout, 200),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(200, 200),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(200, o)
                    ) for o in output_size
                ]
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(hidden_size * lout, 200),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(200, output_size)
            )

        self.nl = nn.SELU()

    def l_out_conv1D(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation *
                 (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def l_out_maxpool1D(self, l_in, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv1D(
            l_in, kernel_size, stride, padding, dilation
        )
        return l_out

    def forward(self, x, noise=None):        
        x = self.preprocess(x)
        if noise is not None:
            x = x + noise
        x = self.batch_norm0(x)

        x = self.dropout(
            self.pool1(
                self.batch_norm1(self.nl(self.conv2(self.nl(self.conv1(x)))))
            )
        )

        x = self.dropout(
            self.pool2(
                self.batch_norm2(self.nl(self.conv4(self.nl(self.conv3(x)))))
            )
        )

        x = self.dropout(
            self.pool3(
                self.batch_norm3(self.nl(self.conv6(self.nl(self.conv5(x)))))
            )
        )

        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(x.view(x.size(0), -1)) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(x.view(x.size(0), -1))

        return pred


class TransformerNet(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 n_layers=2,
                 kernel_size=2,
                 pool_size=2,
                 n_heads=4,
                 key_dim=None,
                 val_dim=None,
                 inner_dim=None,
                 dropout=0.1

                 ):
        super(TransformerNet, self).__init__()
        self.preprocess = Preprocessor()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        # size of output
        lout = self.l_out_conv1d(3750, kernel_size)
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1d(lout, pool_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1d(lout, kernel_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1d(lout, pool_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1d(lout, kernel_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1d(lout, pool_size)

        print('lout: ', lout)

        self.nl = nn.ReLU()

        if key_dim is None:
            key_dim = hidden_size // n_heads

        if val_dim is None:
            val_dim = hidden_size // n_heads

        if inner_dim is None:
            inner_dim = hidden_size // 2

        self.layer_stack = [] if n_layers == 0 else nn.ModuleList([
            EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        if not isinstance(output_size, (list, tuple)):
            output_size = [output_size]

        output_modules = [
            nn.Sequential(
                # EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                # EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                EncoderTaskLayer2(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                nn.Linear(hidden_size, 200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Linear(200, o)
            ) for o in output_size
        ]
        if len(output_modules) == 1:
            self.out = output_modules[0]
        else:
            self.out = nn.ModuleList(output_modules)

    def l_out_conv1d(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation *
                 (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def l_out_maxpool1d(self, l_in, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv1d(
            l_in, kernel_size, stride, padding, dilation
        )
        return l_out

    def forward(self, x, noise=None):
        #x = self.preprocess(x)
        if noise is not None:
            x = x + noise

        x = self.nl(self.pool1(self.conv1(x)))
        x = self.nl(self.pool2(self.conv2(x)))
        x = self.nl(self.pool3(self.conv3(x)))

        data = x.permute(0, 2, 1)

        for enc_layer in self.layer_stack:
            data, _ = enc_layer(data)

        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(data) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(data)

        return pred
    
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # print('q: ', q.size())
        # print('k: ', k.size())
        # print('v: ', v.size())

        # print('n_head: ', self.n_head)
        # print('d_k: ', self.d_k)
        # print('d_v: ', self.d_v)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MultiHeadTaskAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.num_embeddings = 1
        self.d_model = d_model
        self.embedding = nn.Embedding(self.num_embeddings, d_model)
        self.multihead = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)

    def forward(self, q, k, v, mask=None):
        assert type(q) == int
        assert q < self.num_embeddings
        # q = torch.LongTensor([[q]]).expand(k.size(0), 1).to(k.device)
        # q = self.embedding(q)
        q = (torch.ones((k.size(0), 1, self.d_model)) / self.d_model).to(k.device)

        return self.multihead(q, k, v, mask)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1, d_out=None):
        super().__init__()
        if d_out is None:
            d_out = d_in
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_out, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.d_out == self.d_in:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, d_out=None, attn_flag=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, d_out=d_out)
        self.attn_flag = attn_flag

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        if self.attn_flag:
            return enc_output, enc_slf_attn
        else:
            return enc_output


class EncoderTaskLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, d_out=None, attn_flag=True):
        super(EncoderTaskLayer, self).__init__()
        self.slf_attn = MultiHeadTaskAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, d_out=d_out)
        self.attn_flag = attn_flag

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(0, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        if self.attn_flag:
            return enc_output.squeeze(1), enc_slf_attn
        else:
            return enc_output.squeeze(1)


class EncoderTaskLayer2(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, d_out=None, attn_flag=True):
        super(EncoderTaskLayer2, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, d_out=d_out)
        self.attn_flag = attn_flag

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        enc_output = enc_output.sum(dim=1, keepdim=True)

        if self.attn_flag:
            return enc_output.squeeze(1), enc_slf_attn
        else:
            return enc_output.squeeze(1)
        
class Preprocessor(nn.Module):

    def __init__(
            self,
            ma_window_size=2,
            mv_window_size=4,
            num_samples_per_second=125):
        # ma_window_size: (in seconds) window size to use
        #                 for moving average baseline wander removal
        # mv_window_size: (in seconds) window size to use
        #                 for moving average RMS normalization

        super(Preprocessor, self).__init__()

        # Kernel size to use for moving average baseline wander removal: 2
        # seconds * 125 HZ sampling rate, + 1 to make it odd

        self.maKernelSize = (ma_window_size * num_samples_per_second) + 1

        # Kernel size to use for moving average normalization: 4
        # seconds * 125 HZ sampling rate , + 1 to make it odd

        self.mvKernelSize = (mv_window_size * num_samples_per_second) + 1

    def forward(self, x):

        with torch.no_grad():

            # Remove window mean and standard deviation

            x = (x - torch.mean(x, dim=2, keepdim=True)) / \
                (torch.std(x, dim=2, keepdim=True) + 0.00001)

            # Moving average baseline wander removal

            x = x - F.avg_pool1d(
                x, kernel_size=self.maKernelSize,
                stride=1, padding=(self.maKernelSize - 1) // 2
            )

            # Moving RMS normalization

            x = x / (
                torch.sqrt(
                    F.avg_pool1d(
                        torch.pow(x, 2),
                        kernel_size=self.mvKernelSize,
                        stride=1, padding=(self.mvKernelSize - 1) // 2
                    )) + 0.00001
            )

        # Don't backpropagate further

        x = x.detach().contiguous()

        return x

