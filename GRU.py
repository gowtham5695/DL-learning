import torch
import torch.nn as nn

gru=nn.GRU(input_size=4,hidden_size=8,batch_first=True)

x=torch.randn(1,5,4)

output,hidden=gru(x)

print(output)
print(output.shape)
print(hidden)
print(hidden.shape)
