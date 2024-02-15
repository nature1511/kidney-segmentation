import torch


def tta(x, model):
    model.eval()
    x_n = [torch.rot90(x, k=i, dims=(-2, -1)) for i in range(4)]

    shape = x.shape
    for i in range(4):
        if i == 0:
            x_n[i] = x_n[i]
        else:
            x_n[i] = torch.flip(x_n[i], dims=(-2,))
    with torch.no_grad():
        pred = [model(b) for b in x_n]

        torch.flip(pred[i], dims=(-2,))
    pred = torch.cat(pred, dim=0)
    pred = pred.sigmoid()
    pred = pred.reshape(4, shape[0], *shape[2:])

    for i in range(4):
        if i == 0:
            pred[i] = pred[i]
        else:
            pred[i] = torch.flip(pred[i], dims=(-2,))
    pred = [torch.rot90(pred[i], k=-i, dims=(-2, -1)) for i in range(4)]
    pred = torch.stack(pred, dim=0).mean(0)
    return pred
