# torch에서 제공하는 optimizer
from torch.optim import Adam,SGD,Adadelta,AdamW
# 그외  https://github.com/jettify/pytorch-optimizer
from torch_optimizer import adamp,radam
import madgrad
_optimizer_dict={
        'AdamP' :adamp.AdamP,
        'Adam' : Adam,
        'Adadelta':Adadelta,
        'AdamW':AdamW,
        'SGD' : SGD,
        'madgrad' : madgrad.MADGRAD,
        'radam' : radam.RAdam
    }
def optimizer_entrypoint(optimizer_name):
    return _optimizer_dict[optimizer_name]

def create_optimizer(optimizer_name,model,lr):
    
    create_fn = optimizer_entrypoint(optimizer_name)
    if optimizer_name=='AdamP':
        optimizer = create_fn(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,betas=(0.9, 0.999), weight_decay=1e-2)
    elif optimizer_name=='Adam':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay=0)
    elif optimizer_name=='Adadelta':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay=0)
    elif optimizer_name=='AdamW':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay=0.01)
    elif optimizer_name=='SGD':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay=0)
    elif optimizer_name=='madgrad':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, momentum = 0.9, weight_decay = 0, eps = 1e-06)
    elif optimizer_name=='radam':
        optimizer = create_fn(filter(lambda p: p.requires_grad,model.parameters()),lr=lr, weight_decay = 0)
    return  optimizer

if __name__ == "__main__":
    from torchvision import models
    model=models.resnet18()
    create_optimizer('AdamP',model,0.001)