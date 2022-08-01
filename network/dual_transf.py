import torch.nn as nn
import torch
import numpy as np
from single_trans import TransformerEncoderLayer
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

class Dual_Transformer(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()

        self.input_size = input_size
        # print(input_size,output_size)
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            # self.row_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            # self.col_trans.append(TransformerEncoderLayer(d_model=input_size//2, nhead=4, dropout=dropout, bidirectional=True))
            # self.row_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))
            # self.col_norm.append(nn.GroupNorm(1, input_size//2, eps=1e-8))原代码

            self.row_trans.append(TransformerEncoderLayer(d_model=input_size, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(TransformerEncoderLayer(d_model=input_size, nhead=4, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))


        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    # nn.Conv2d(input_size//2, output_size, 1) 原代码
                                    nn.Conv2d(input_size // 2, input_size, 1)
                                    )

    def forward(self, input):
        # print("input:",input.shape)

        # 为了把分割之后对应的维度 与transformer匹配，需要把K和S交换一下 [B:batch size, N:特征维度， K：块长， S：块数]
        # num_frame 对应 块数S，frame_size 对应 块长K

        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape #原代码
        # b, c, dim2, dim1 = input_1.shape
        # output = self.input(input)
        output = input
        # print("output.shape",output.shape)
        for i in range(len(self.row_trans)):
            # row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)  # [dim1, b*dim2, c] 原代码
            # 根据dprnn的 rnn修改   input: [B, N, K, S]  # intra RNN   [BS, K, N]
            row_input = output.permute(0, 3, 2, 1).contiguous().view(b * dim1, dim2, -1)  # [b*dim1, dim2, c]
            # print("row_input",row_input.shape)
            row_output = self.row_trans[i](row_input) # 新代码[b*dim1, dim2, c]    # 原代码[dim1, b*dim2, c]
            # row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [b, c, dim2, dim1] 原代码
            row_output = row_output.view(b, dim1, dim2, -1).permute(0, 3, 2, 1).contiguous()  # [b, c, dim2, dim1]
            row_output = self.row_norm[i](row_output)  # [b, c, dim2, dim1]
            output = output + row_output  # [b, c, dim2, dim1]

            # inter RNN   [BK, S, N]
            #
            # col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)  # [dim2, b*dim1, c] 原代码
            col_input = output.permute(0, 2, 3, 1).contiguous().view(b * dim2, dim1, -1)  # [b*dim2, dim1, c]
            col_output = self.col_trans[i](col_input)  # [b*dim2, dim1, c]
            # col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [b, c, dim2, dim1] 原代码
            col_output = col_output.view(b, dim2, dim1, -1).permute(0, 3, 1, 2).contiguous()  # [b, c, dim2, dim1]
            col_output = self.col_norm[i](col_output)  # [b, c, dim2, dim1]
            output = output + col_output  # [b, c, dim2, dim1]

        del row_input, row_output, col_input, col_output
        # print("output1.shape",output.shape)
        # output = self.output(output)  # [b, c, dim2, dim1] 原代码
        # print("output2.shape", output.shape)

        return output

''''''


trans = Dual_Transformer(64, 64, num_layers=4)
print(trans)
trans = torch.nn.DataParallel(trans)
# trans = trans.cuda()
src = torch.rand(2, 64, 250, 8)
out = trans(src)
print(out.shape)


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num

print(numParams(trans))


