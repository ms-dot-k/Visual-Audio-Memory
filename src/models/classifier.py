import torch
from torch import nn

class GRU_Backend(nn.Module):
    def __init__(self, logits=True):
        super().__init__()
        self.logits = logits

        self.gru = nn.GRU(512, 512, 2, bidirectional=True, dropout=0.3)
        if logits:
            self.fc = nn.Linear(1024, 500)

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()  # S,B,512
        self.gru.flatten_parameters()

        x, _ = self.gru(x)
        x = x.mean(0, keepdim=False)

        if self.logits:
            pred = self.fc(x)    # B, 500
            return pred
        else:
            return x

##### MS-TCN from https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Network

class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()

class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d( n_inputs, n_inputs, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d( n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding,True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv( x )
            out = self.batchnorm( out )
            out = self.chomp( out )
            return self.non_lin( out )

class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2,
                 relu_type = 'relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_kernels = len( kernel_sizes )
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx ,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr( self ,'cbcr0_{}'.format(k_idx), cbcr )
        self.dropout0 = nn.Dropout(dropout)

        for k_idx ,k in enumerate( self.kernel_sizes ):
            cbcr = ConvBatchChompRelu( n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type, dwpw=dwpw)
            setattr( self ,'cbcr1_{}'.format(k_idx), cbcr )
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs//self.num_kernels) != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self ,'cbcr0_{}'.format(k_idx))
            outputs.append( branch_convs(x) )
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0( out0 )

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range( self.num_kernels ):
            branch_convs = getattr(self ,'cbcr1_{}'.format(k_idx))
            outputs.append( branch_convs(out0) )
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1( out1 )

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            padding = [ (s-1)*dilation_size for s in self.ksizes]
            layers.append( MultibranchTemporalBlock( in_channels, out_channels, self.ksizes,
                stride=1, dilation=dilation_size, padding = padding, dropout=dropout, relu_type = relu_type,
                dwpw=dwpw) )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# --------------------------------

class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = out.mean(2)
        return self.tcn_output(out)

class Temp_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        tcn_options = {"dropout": 0.2, "dwpw": False, "kernel_size": [3,5,7], "num_layers": 4, "width_mult": 1}

        tcn_class = MultiscaleMultibranchTCN
        self.tcn = tcn_class(input_size=512,
                             num_channels=[256 * len([3,5,7]) * 1] * 4,
                             num_classes=500,
                             tcn_options=tcn_options,
                             dropout=tcn_options['dropout'],
                             relu_type='prelu',
                             dwpw=tcn_options['dwpw'],
                             )

    def forward(self, x):
        x = self.tcn(x)
        return x