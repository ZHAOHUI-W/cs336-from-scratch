import torch


def save_checkpoint(model, optimizer, iteration, out):
    save_dict = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "iteration":iteration}

    torch.save(save_dict,out)




def load_checkpoint(src, model, optimizer):
    save_dict = torch.load(src)

    model.load_state_dict(save_dict["model"])
    optimizer.load_state_dict(save_dict["optimizer"])

    return save_dict["iteration"]
