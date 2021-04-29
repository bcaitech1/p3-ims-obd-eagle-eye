import torch.optim
from torch.optim import adam,SGD
from adamp import AdamP
_optimizer_dict={
        'AdamP' :AdamP,
        'Adam' : adam,
        'SGD' : SGD,
    }
def optimizer_entrypoint(optimizer_name):
    return _optimizer_dict[optimizer_name]

def create_optimizer(optimizer_name,model,lr):
    
    create_fn = optimizer_entrypoint(optimizer_name)
    if optimizer_name=='AdamP':
        optimizer = create_fn(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,betas=(0.9, 0.999), weight_decay=1e-2)
    elif optimizer_name=='Adam':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay=1e-6)
    elif optimizer_name=='SGD':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay=1e-6)
    return  optimizer