from tqdm import tqdm


def trainer(
    args,
    model,
    device,
    loader, 
    optimiser,
    loss,
    scheduler
):
    model.train()
    train_loss = 0
    for _, batch in enumerate(tqdm(loader)):
        if args.model == '3dcnn':
            x = batch.to(device).float().squeeze(0)
        else:
            x = batch.to(device).float().squeeze(0)
            x = x.view(x.shape[0] * x.shape[2], x.shape[1], x.shape[3], x.shape[4])
        optimiser.zero_grad()
        x_cap, _ = model(x)
        l = loss(x, x_cap)
        l.backward()
        optimiser.step()
        train_loss += l.item()
    scheduler.step()
    return train_loss / len(loader)

