import torch 

x=torch.tensor(2.0)
w1=torch.tensor(0.5,requires_grad=True)
w2=torch.tensor(-0.3,requires_grad=True)
v1=torch.tensor(0.8,requires_grad=True)
v2=torch.tensor(0.6,requires_grad=True)
b1=torch.tensor(0.0,requires_grad=True)
b2=torch.tensor(0.0,requires_grad=True)
b_out=torch.tensor(0.0,requires_grad=True)
Y_true=torch.tensor(1.0)

h1=w1*x+b1
h2=w2*x+b2
y_pred=v1*h1+v2*h2+b_out
loss=(y_pred-Y_true)**2

print("Y_pred:", y_pred.item())
print("Loss:", loss.item())

loss.backward()

print("Gradients:")
print("dl/w1:",w1.grad.item())
print("dl/dw2:",w2.grad.item())
print("dl/dv1=",v1.grad.item())
print("dl/dv2=",v2.grad.item())
