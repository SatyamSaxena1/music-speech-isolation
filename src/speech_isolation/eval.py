import torch
from .metrics import si_sdr


def evaluate(model, dataloader, device='cpu'):
    model.eval()
    scores = []
    with torch.no_grad():
        for x in dataloader:
            # x: mixture
            pred_speech, pred_bg = model(x)
            # compute metrics against oracle if available
            scores.append(0.0)
    # return average metrics dict
    return {'si_sdr': sum(scores)/len(scores) if scores else 0.0}
