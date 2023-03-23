import os
import sys
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR



class Trainer(object):
    def __init__(
        self, args, device, model
    ):
        self.args = args
        self.model = model
        self.device = device

        self.ep_idx = 0
        self.pre_trained = False
        self.idx = 0
        self.train_size = 0
        self.val_size = 0

        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.gamma = self.args.gamma
        self.epochs = self.args.epochs

        self.adv_update = False

        if self.args.type == 'multi':
            model_params_main = \
                list(self.model.dense1.parameters()) + \
                list(self.model.dense2.parameters()) + \
                list(self.model.dense3.parameters()) + \
                list(self.model.enc1.parameters()) + \
                list(self.model.enc2.parameters()) + \
                list(self.model.pred_fc1.parameters()) + \
                list(self.model.pred_fc2.parameters()) + \
                list(self.model.emb1.parameters()) + \
                list(self.model.emb2.parameters()) + \
                list(self.model.dec1.parameters()) + \
                list(self.model.dec2.parameters()) + \
                list(self.model.dec3.parameters())   
        else:
            model_params_main = \
                list(self.model.enc1.parameters()) + \
                list(self.model.enc2.parameters()) + \
                list(self.model.pred_fc1.parameters()) + \
                list(self.model.pred_fc2.parameters()) + \
                list(self.model.emb1.parameters()) + \
                list(self.model.emb2.parameters()) + \
                list(self.model.dec1.parameters()) + \
                list(self.model.dec2.parameters()) + \
                list(self.model.dec3.parameters())

        self.optim_main = Adam(
            model_params_main,
            lr = float(self.args.main_lr),
            weight_decay = float(self.args.decay)
        )
        # self.optim_main = SGD(
        #     model_params_main,
        #     lr = self.args.adv_lr,
        #     weight_decay = self.args.decay
        # )


        model_params_pred = \
            list(self.model.pred_final.parameters())

        self.optim_pred = Adam(
            model_params_pred,
            lr = 1e-3,
            weight_decay = float(self.args.decay)
        )
        # self.optim_pred = SGD(
        #     model_params_pred,
        #     lr = 1e-3,
        #     weight_decay = self.args.decay
        # )

        model_params_adv = \
            list(self.model.dis_1to2.parameters()) + \
            list(self.model.dis_2to1.parameters()) 
        self.optim_adv = Adam(
            model_params_adv,
            lr = float(self.args.adv_lr),
            weight_decay = float(self.args.decay)
        )
        # self.optim_adv = SGD(
        #     model_params_adv,
        #     lr = self.args.adv_lr,
        #     weight_decay = float(self.args.decay)
        # )

        # if self.args.lr_decay:
        #     self.scheduler_main = StepLR(
        #         self.optim_main,


        self.criterion_pred = nn.CrossEntropyLoss()
        self.criterion_recon = nn.MSELoss()
        self.criterion_dis = nn.MSELoss()



    def forward_pass_main(
        self, x, y, x1 = None
    ):
        if x1 is not None:
            y_hat, x_hat, e1, e2 = self.model.forward_main(x, x1)
        else:
            y_hat, x_hat, e1, e2 = self.model.forward_main(x)
        loss_pred = self.criterion_pred(y_hat, y)
        loss_recon = self.criterion_recon(x_hat, x)
        total_loss = self.alpha * loss_pred + self.beta * loss_recon
        return total_loss, y_hat, x_hat, e1, e2



    def forward_pass_adv(
        self, e1, e2, adv_update = True
    ):

        e1_hat, e2_hat = self.model.forward_adv(e1, e2)
        if adv_update:
            loss_dis_1to2 = self.criterion_dis(e2_hat, e2)
            loss_dis_2to1 = self.criterion_dis(e1_hat, e1)
        else:
            e1_rand = (-2 * torch.rand(
                            e1.shape[0], e1.shape[1]) + 1
                            ).to(self.device)
            e2_rand = (-2 * torch.rand(
                            e2.shape[0], e2.shape[1]) + 1
                            ).to(self.device)
            loss_dis_1to2 = self.criterion_dis(
                            e2_hat, e2_rand)
            loss_dis_2to1 = self.criterion_dis(
                            e1_hat, e1_rand)

        loss_adv = loss_dis_1to2 + loss_dis_2to1
        return loss_adv



    def save_model(
        self, ckpt
    ):
        state = {
            'epoch': self.ep_idx,
            'model_state_dict': self.model.state_dict(),
            'optimiser_main': self.optim_main.state_dict(),
            'optimiser_adv': self.optim_adv.state_dict(),
        }
        torch.save(state, ckpt)



    def fit(
        self, train, val
    ):

        self.train_size = len(train)
        self.val_size = len(val)

        self.model.train().to(self.device)

        if self.ep_idx >= self.epochs - 1:
            print(f"Training has been completed for {self.epochs} epochs.")
            sys.exit(1)

        best_val_accuracy = 0.0
        best_ep = self.ep_idx

        for epoch in range(self.ep_idx, self.epochs):
            self.ep_idx = epoch
            true_spks_labels = []
            pred_spks_labels = []
            for batch_idx, (feat, labels) in tqdm(enumerate(train)):
                self.idx = batch_idx

                if self.args.type == 'multi':
                    feat, feat1 = feat
                    feat = feat.type(torch.FloatTensor).to(self.device)
                    feat1 = feat1.type(torch.FloatTensor).to(self.device)
                else:
                    feat = feat.type(torch.FloatTensor).to(self.device)

                labels = labels.to(self.device)
                self.optim_main.zero_grad()
                self.optim_pred.zero_grad()
                self.optim_adv.zero_grad()

                if self.args.type == 'multi':
                    loss_main, y_hat, x_hat, e1, e2 \
                        = self.forward_pass_main(feat, labels, feat1)
                else:
                    loss_main, y_hat, x_hat, e1, e2 \
                        = self.forward_pass_main(feat, labels)

                if batch_idx % (self.args.num_adv_updates + 1) == 0:
                    self.adv_update = False
                    loss_adv = self.forward_pass_adv(e1, e2, self.adv_update)
                    total_loss = loss_main + self.gamma * loss_adv
                    total_loss.backward()
                    self.optim_main.step()
                    self.optim_pred.step()
                else:
                    self.adv_update = True
                    loss_adv = self.forward_pass_adv(e1, e2, self.adv_update)
                    loss_adv.backward()
                    self.optim_adv.step()
                
                true_spks_labels.append(labels.data.cpu().numpy())
                pred_spks_labels.append(torch.argmax(y_hat, axis = 1).data.cpu().numpy().tolist())   

            true_spks_labels = [yy for y in true_spks_labels for yy in y]   
            pred_spks_labels = [yy for y in pred_spks_labels for yy in y]

            train_accuracy = accuracy_score(true_spks_labels, pred_spks_labels)

            print(f"Training results:")
            # print(f"True labels: {true_spks_labels}")
            # print(f"Predicted labels: {pred_spks_labels}")

            print(f"Performing validation for epoch {epoch}")

            true_spks_labels = []
            pred_spks_labels = []

            self.model.eval().to(self.device)


            with torch.no_grad():
                # if self.args.save_val_embeddings:
                #     val_e1 = []
                #     val_e2 = []

                for batch_idx, (feat, labels) in tqdm(enumerate(val)):
                    self.idx = batch_idx
                    if self.args.type == 'multi':
                        feat, feat1 = feat
                        feat = feat.type(torch.FloatTensor).to(self.device)
                        feat1 = feat1.type(torch.FloatTensor).to(self.device)
                    else:
                        feat = feat.type(torch.FloatTensor).to(self.device)
                    labels = labels.to(self.device)

                    if self.args.type == 'multi':
                        loss_main, y_hat, x_hat, e1, e2 \
                            = self.forward_pass_main(feat, labels, feat1)
                    else:
                        loss_main, y_hat, x_hat, e1, e2 \
                            = self.forward_pass_main(feat, labels)

                    loss_adv = self.forward_pass_adv(e1, e2)

                    true_spks_labels.append(labels.data.cpu().numpy())
                    pred_spks_labels.append(torch.argmax(y_hat, axis = 1).data.cpu().numpy().tolist())

                    # if self.args.save_val_embeddings:
                    #     val_e1.append(e1.data.cpu().numpy())
                    #     val_e2.append(e2.data.cpu().numpy())

                # val_e1 = np.array([x for y in val_e1 for x in y], dtype = float)
                # val_e2 = np.array([x for y in val_e2 for x in y], dtype = float) 

                true_spks_labels = [yy for y in true_spks_labels for yy in y]
                pred_spks_labels = [yy for y in pred_spks_labels for yy in y]

                print("Validation results:")
                # print(f"True labels: {true_spks_labels}")
                # print(f"Predicted labels: {pred_spks_labels}")

                val_acc = accuracy_score(true_spks_labels, pred_spks_labels)

                # if self.args.save_val_embeddings:
                #     if val_acc > best_val_accuracy:
                        # np.save(
                        #     os.path.join(self.args.out_dir, 'val_e1.npy'),
                        #     val_e1
                        # )
                        # np.save(
                        #     os.path.join(self.args.out_dir, 'val_e2.npy'),
                        #     val_e2
                        # )
                        # np.save(
                        #     os.path.join(self.args.out_dir, 'val_true_spks_labels.npy'),
                        #     true_spks_labels
                        # )
                        # with open(
                        #     os.path.join(self.args.out_dir, 'best_epoch.txt'),
                        #     'w'
                        # ) as f:
                        #     f.write(f"Best epoch : {str(epoch)}")
                        # pass

                # best_epoch_flag = False
                ckpt_loc = f"{self.args.ckpt}/{self.args.type}"
                if not os.path.exists(ckpt_loc):
                    os.makedirs(ckpt_loc)
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_ep = epoch
                    best_epoch_flag = True 
                    if self.args.data_augmentation:
                        save_path = os.path.join(ckpt_loc,
                        f'epoch_{self.args.type}_{self.args.pre_model}_{self.args.evaluation}_aug.pth')
                    else:
                        save_path = os.path.join(ckpt_loc,
                        f'epoch_{self.args.type}_{self.args.pre_model}.pth')
                    self.save_model(save_path)
                    print(f"Best epoch: {best_ep} | Best val accuracy: {best_val_accuracy:.4f}")

                # if self.ep_idx % 10 == 0 or best_epoch_flag:
                # if best_epoch_flag:
                #     ckpt = os.path.join(
                #         ckpt_loc, f'epoch_{epoch}_{self.args.pre_model}.pt'
                #     )
                #     self.save_model(ckpt)

                print(
                    f"Epoch {epoch} | "
                    f"Train Accuracy: {train_accuracy:.4f} | "
                    f"Val Accuracy: {val_acc:.4f} | "
                )

        return best_ep, best_val_accuracy