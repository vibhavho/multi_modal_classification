import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Inference(object):
    def __init__(
        self, args, device, model
    ):
        self.args = args
        self.model = model
        self.device = device

        self.idx = 0


    def forward_pass_main(
        self, x, y, x1 = None
    ):
        if x1 is not None:
            y_hat, x_hat, e1, e2 = self.model.forward_main(x, x1)
        else:
            y_hat, x_hat, e1, e2 = self.model.forward_main(x)
        return y_hat, x_hat, e1, e2


    def evaluate(self, test):
        self.model.eval().to(self.device)
        true_spks_labels = []
        pred_spks_labels = []

        with torch.no_grad():
            # if self.args.save_test_embeddings:
            #     test_e1 = []
            #     test_e2 = []
            for batch_idx, (feat, labels) in tqdm(enumerate(test)):
                self.idx = batch_idx
                if self.args.type == 'multi':
                    feat, feat1 = feat
                    feat = feat.type(torch.FloatTensor).to(self.device)
                    feat1 = feat1.type(torch.FloatTensor).to(self.device)
                else:
                    feat = feat.type(torch.FloatTensor).to(self.device)
                labels = labels.to(self.device)
                if self.args.type == 'multi':
                    y_hat, x_hat, e1, e2 \
                            = self.forward_pass_main(feat, labels, feat1)
                else:
                    y_hat, x_hat, e1, e2 \
                        = self.forward_pass_main(feat, labels)

                true_spks_labels.append(labels.data.cpu().numpy())
                pred_spks_labels.append(torch.argmax(y_hat, axis = 1).data.cpu().numpy().tolist())

                # if self.args.save_test_embeddings:
                #     test_e1.append(e1.data.cpu().numpy())
                #     test_e2.append(e2.data.cpu().numpy())

            # test_e1 = np.array([x for y in test_e1 for x in y], dtype = float)
            # test_e2 = np.array([x for y in test_e2 for x in y], dtype = float)

            true_spks_labels = [yy for y in true_spks_labels for yy in y]
            pred_spks_labels = [yy for y in pred_spks_labels for yy in y]

            test_acc = accuracy_score(true_spks_labels, pred_spks_labels)

            # print(f"True labels: {true_spks_labels}")
            # print(f"Predicted labels: {pred_spks_labels}")


        return test_acc