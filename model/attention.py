import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, lambda_val):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.lambda_val = lambda_val

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        attn = torch.bmm(output, context.transpose(1, 2))
        scaled_attn = attn * self.lambda_val
        attn = F.softmax(scaled_attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        
        return torch.sum(output, dim=1, keepdim=True), attn
    
