import torch
import torch.nn as nn

lstm=nn.LSTM(input_size=4,hidden_size=8,batch_first=True)

x=torch.randn(1,5,4)

output,(h,cell_state)=lstm(x)

print(output)
print(output.shape)
print(h)
print(h.shape)
print(cell_state)
print(cell_state.shape)