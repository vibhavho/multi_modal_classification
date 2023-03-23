import torch
import torch.nn as nn


class UAI(nn.Module):
    def __init__(
        self, model_config, train_args
    ):
        super(UAI, self).__init__()
        self.args = model_config
        self.train_args = train_args
        self.noisy_drop_rate = self.args.dropout
        self.embedding_dim_1 = self.args.embed_dim1
        self.embedding_dim_2 = self.args.embed_dim2

                
    
    def init_model(self):
        if self.train_args.type == 'multi':
            self.dense1 = nn.Sequential(
                nn.Linear(self.args.x_shape[0], 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )

            self.dense2 = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )

            self.dense3 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Tanh()
            )
            
        self.enc1 = nn.Sequential(
            nn.Linear(self.args.x_shape[0], 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.emb1 = nn.Sequential(
            nn.Linear(512, self.embedding_dim_1),
            nn.Tanh()
        )
        self.emb2 = nn.Sequential(
            nn.Linear(512, self.embedding_dim_2),
            nn.Tanh()
        )
        self.pred_fc1 = nn.Sequential(
            nn.BatchNorm1d(self.embedding_dim_1),
            nn.Linear(self.embedding_dim_1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pred_fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pred_final = nn.Sequential(
            nn.Linear(512, self.args.spks),
            nn.Softmax(dim = 1)
        )

        self.noisy_transformer = nn.Sequential(
            nn.Dropout(self.noisy_drop_rate)
        )

        final_embed_dim = self.embedding_dim_1 + self.embedding_dim_2

        self.dec1 = nn.Sequential(
            nn.Linear(final_embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.dec3 = nn.Sequential(
            nn.Linear(512, self.args.x_shape[0]),
            nn.Sigmoid()
        )

        self.dis_1to2 = nn.Sequential(
            nn.Linear(self.embedding_dim_1, self.embedding_dim_1),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(self.embedding_dim_1, self.embedding_dim_1),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(self.embedding_dim_1, self.embedding_dim_2),
            nn.Tanh()

        )

        self.dis_2to1 = nn.Sequential(
            nn.Linear(self.embedding_dim_2, self.embedding_dim_2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(self.embedding_dim_2, self.embedding_dim_2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(self.embedding_dim_2, self.embedding_dim_1),
            nn.Tanh()
        )



    def init_weights(self):
        if self.train_args.type == 'multi':
            torch.nn.init.xavier_uniform_(self.dense1[0].weight)
            torch.nn.init.xavier_uniform_(self.dense2[0].weight)
            torch.nn.init.orthogonal_(self.dense3[0].weight)
        torch.nn.init.xavier_uniform_(self.enc1[0].weight)
        torch.nn.init.xavier_uniform_(self.enc2[0].weight)
        torch.nn.init.orthogonal_(self.emb1[0].weight)
        torch.nn.init.orthogonal_(self.emb2[0].weight)
        torch.nn.init.xavier_uniform_(self.pred_fc1[1].weight)
        torch.nn.init.xavier_uniform_(self.pred_fc2[0].weight)
        torch.nn.init.xavier_uniform_(self.pred_final[0].weight)
        torch.nn.init.xavier_uniform_(self.dec1[0].weight)
        torch.nn.init.xavier_uniform_(self.dec2[0].weight)
        torch.nn.init.xavier_uniform_(self.dec3[0].weight)
        torch.nn.init.xavier_uniform_(self.dis_1to2[0].weight)
        torch.nn.init.xavier_uniform_(self.dis_1to2[3].weight)
        torch.nn.init.xavier_uniform_(self.dis_1to2[6].weight)
        torch.nn.init.xavier_uniform_(self.dis_2to1[0].weight)
        torch.nn.init.xavier_uniform_(self.dis_2to1[3].weight)
        torch.nn.init.xavier_uniform_(self.dis_2to1[6].weight)

        if self.train_args.type == 'multi':
            if self.dense1[0].bias is not None:
                torch.nn.init.zeros_(self.dense1[0].bias)
            if self.dense2[0].bias is not None:
                torch.nn.init.zeros_(self.dense2[0].bias)
            if self.dense3[0].bias is not None:
                torch.nn.init.zeros_(self.dense3[0].bias)
        if self.enc1[0].bias is not None:
            torch.nn.init.zeros_(self.enc1[0].bias)
        if self.enc2[0].bias is not None:
            torch.nn.init.zeros_(self.enc2[0].bias)
        if self.emb1[0].bias is not None:
            torch.nn.init.zeros_(self.emb1[0].bias)
        if self.emb2[0].bias is not None:
            torch.nn.init.zeros_(self.emb2[0].bias)
        if self.pred_fc1[1].bias is not None:
            torch.nn.init.zeros_(self.pred_fc1[1].bias)
        if self.pred_fc2[0].bias is not None:
            torch.nn.init.zeros_(self.pred_fc2[0].bias)
        if self.pred_final[0].bias is not None:
            torch.nn.init.zeros_(self.pred_final[0].bias)
        if self.dec1[0].bias is not None:
            torch.nn.init.zeros_(self.dec1[0].bias)
        if self.dec2[0].bias is not None:
            torch.nn.init.zeros_(self.dec2[0].bias)
        if self.dec3[0].bias is not None:
            torch.nn.init.zeros_(self.dec3[0].bias)
        if self.dis_1to2[0].bias is not None:
            torch.nn.init.zeros_(self.dis_1to2[0].bias)
        if self.dis_1to2[3].bias is not None:
            torch.nn.init.zeros_(self.dis_1to2[3].bias)
        if self.dis_1to2[6].bias is not None:
            torch.nn.init.zeros_(self.dis_1to2[6].bias)
        if self.dis_2to1[0].bias is not None:
            torch.nn.init.zeros_(self.dis_2to1[0].bias)
        if self.dis_2to1[3].bias is not None:
            torch.nn.init.zeros_(self.dis_2to1[3].bias)
        if self.dis_2to1[6].bias is not None:
            torch.nn.init.zeros_(self.dis_2to1[6].bias)


    def multi_concat(self, x1, x2):
        x1_cap = self.dense1(x1)
        x2_cap = self.dense2(x2)
        x = torch.cat((x1_cap, x2_cap), 1)
        x = self.dense3(x)
        return x


    def encoder(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        e1 = self.emb1(x)
        e2 = self.emb2(x)
        return e1, e2


    def decoder(self, x):
        e1_prime, e2 = x
        x = torch.cat((e1_prime, e2), 1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x


    def predict(self, x):
        x = self.pred_fc1(x)
        x = self.pred_fc2(x)
        x = self.pred_final(x)
        return x
    

    def forward_main(self, x, x2 = None):
        if x2 is not None:
            x = self.multi_concat(x, x2)

        e1, e2 = self.encoder(x)
        e1_prime = self.noisy_transformer(e1)

        x_hat = self.decoder((e1_prime, e2))
        y_hat = self.predict(e1)

        return y_hat, x_hat, e1, e2


    def forward_adv(self, e1, e2):
        e2_hat = self.dis_1to2(e1)
        e1_hat = self.dis_2to1(e2)
        return e1_hat, e2_hat