import os
import torch



def save_model(
    epochs, model, optimizer, criterion, ckpt, aug, model_name = None
):
    if aug:
        save_loc = f"{ckpt}/{model_name}_aug"
    else:
        save_loc = f"{ckpt}/{model_name}"
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, os.path.join(save_loc, 'final_model.pth')
    )


class SaveBestModel:
    def __init__(
        self, best_loss = float('inf')
    ):
        self.best_loss = best_loss
        
    def __call__(
        self, loss, epoch, model, optimizer, criterion, ckpt, aug, model_name = None
    ):  
        if aug:
            save_loc = f"{ckpt}/{model_name}_aug"
        else:
            save_loc = f"{ckpt}/{model_name}"
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"\nSaving best model for epoch: {epoch}\n")

            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
                
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(save_loc, 'best_model.pth')
            )

