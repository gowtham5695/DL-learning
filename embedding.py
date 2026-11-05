import torch
import torch.nn as nn

"""
embedding=nn.Embedding(10,4)

words=torch.tensor([1,2,3])
#only three rows
output=embedding(words)
#all rows
print(embedding.weight)
print(output)

print(output.shape)"""

"""vocab={
    "cat": 0,
    "dog": 1,
    "apple": 2,
    "king": 3,
    "queen": 4
}
embedding=nn.Embedding(len(vocab),4)

sentence=["cat","dog","king"]

word=torch.tensor([vocab[i] for i in sentence])

out=embedding(word)
print(out)"""


#cosine similarity
import torch.nn.functional as F
d1=torch.tensor([0.1,0.2,0.3])
d2=torch.tensor([0.2,0.1,0.3])
similarity=F.cosine_similarity(d1,d2,dim=0)

print(similarity.item())
