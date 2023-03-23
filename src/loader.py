import os
import numpy as np

import torch 
from torch.utils.data import Dataset,\
                            DataLoader


def lookup_user(id_str):     
    # Function to get user id from folder name
    user_lst = ['fadg0', 'faks0', 'fcft0', 'fcmh0', 'fcmr0', 'fcrh0',
               'fdac1', 'fdms0', 'fdrd1', 'fedw0', 'felc0', 'fgjd0',
               'fjas0', 'fjem0', 'fjre0', 'fjwb0', 'fkms0', 'fpkt0', 'fram1']     
    # can append to list once we add more users     
    return user_lst.index(id_str)


def unisonShuffleDataset(
    spk_ids, xf = None, xs = None
):
    if xs is None:
        p = np.random.permutation(len(xf))
        return xf[p], None, spk_ids[p]
    if xf is None:
        p = np.random.permutation(len(xs))
        return None, xs[p], spk_ids[p]
    else:
        p = np.random.permutation(len(xf))
        return xf[p], xs[p], spk_ids[p]



def loader(args):
    sub_paths = list(os.listdir(args.data))
    xf, xs = [], []
    spk_ids = []

    for sub in sorted(sub_paths):
        cnt = []
        if args.type == 'visual':
            for feat in sorted(os.listdir(f"{args.data}/{sub}/video")):
                if args.data_augmentation:
                    if  '_'.join(feat.split('_')[1:]) == f"{args.pre_model}_aug.npy":
                        vis_feat = np.load(f"{args.data}/{sub}/video/{feat}")
                    else:
                        continue
                else:
                    if  '_'.join(feat.split('_')[1:]) == f"{args.pre_model}.npy":
                        vis_feat = np.load(f"{args.data}/{sub}/video/{feat}")
                    else:
                        continue
                # vis_feat = np.load(f"{args.data}/{sub}/video/{feat.split('_')[0]}_{args.pre_model}.npy")
                # vis_feat = feat if feat.split('_')[-1] == f"{args.pre_model}.npy" else None
                # xf.append(np.mean(np.load(f"{args.data}/{sub}/video/{vis_feat}"), axis = 0))
                # spk_ids += [lookup_user(sub)]
                xf.extend(vis_feat)
                spk_ids += [lookup_user(sub)] * len(vis_feat)
        elif args.type == 'xvec':
            for xv in sorted(os.listdir(f"{args.data}/{sub}/audio")):
                xs.append(np.load(f"{args.data}/{sub}/audio/{xv}"))
                spk_ids += [lookup_user(sub)]
        elif args.type == 'multi':
            for feat in sorted(os.listdir(f"{args.data}/{sub}/video")):
                if args.data_augmentation:
                    if '_'.join(feat.split('_')[1:]) == f"{args.pre_model}_aug.npy":
                        vis_feat = np.load(f"{args.data}/{sub}/video/{feat}")
                    else:
                        continue
                else:
                    if '_'.join(feat.split('_')[1:]) == f"{args.pre_model}.npy":
                        vis_feat = np.load(f"{args.data}/{sub}/video/{feat}")
                    else:
                        continue
                # vis_feat = feat if feat.split('_')[-1] == f"{args.pre_model}.npy" else None
                # vis_emb = np.load(f"{args.data}/{sub}/video/{vis_feat}")
                cnt.append(len(vis_feat))
                xf.extend(vis_feat)
            for idx, xv in enumerate(sorted(os.listdir(f"{args.data}/{sub}/audio"))):
                emb = np.load(f"{args.data}/{sub}/audio/{xv}")
                xs.extend([emb] * cnt[idx])
            spk_ids += [lookup_user(sub)] * sum(cnt)
        else:
            raise Exception("Invalid type of data integration.\
                            Please select one from visual, xvec, or multi.")

    assert len(spk_ids) == len(xf) or len(spk_ids) == len(xs), \
           "Mismatch in data and speaker IDs."

    return np.array(xf), np.array(xs), np.array(spk_ids)



class VidTimit(Dataset):
    def __init__(
        self, args, xf = None, xs = None, spk_ids = None
    ):
        self.xf = xf
        self.xs = xs
        self.args = args
        self.spk_ids = spk_ids

    def __len__(self):
        return len(self.spk_ids)

    def __getitem__(self, idx):
        if self.args.type == 'visual':
            return torch.from_numpy(self.xf[idx]).float(), \
                   self.spk_ids[idx]
        elif self.args.type == 'xvec':
            return torch.from_numpy(self.xs[idx]).float(), \
                   self.spk_ids[idx]
        elif self.args.type == 'multi':
            return (torch.from_numpy(self.xf[idx]).float(), \
                   torch.from_numpy(self.xs[idx]).float()), \
                   self.spk_ids[idx]
        else:
            raise Exception("Invalid type of data integration.\
                            Please select one from visual, xvec, or multi.")




def dataloader(args):
    xf, xs, spk_ids = loader(args)
    assert len(spk_ids) == len(xf) or len(spk_ids) == len(xs), \
              "Mismatch in data and speaker IDs."

    tr_xf, tr_xs = None, None
    te_xf, te_xs = None, None
    val_xf, val_xs = None, None
    

    if args.evaluation == 'seen':
        if args.type == 'visual':
            xf, xs, spk_ids = unisonShuffleDataset(
                spk_ids, xf = xf
            )
        elif args.type == 'xvec':
            xf, xs, spk_ids = unisonShuffleDataset(
                spk_ids, xs = xs
            )
        elif args.type == 'multi':
            xf, xs, spk_ids = unisonShuffleDataset(
                spk_ids, xf = xf, xs = xs
            )
        else:
            raise Exception("Invalid type of data integration.\
                            Please select one from visual, xvec, or multi.")


    elif args.evaluation == 'unseen':
        if args.type == 'visual': xs = None
        elif args.type == 'xvec': xf = None
        elif args.type == 'multi': pass
        else:
            raise Exception("Invalid type of data integration.\
                            Please select one from visual, xvec, or multi.")

    else:
        raise Exception("Invalid evaluation type.\
                        Please select one from seen or unseen.")
        
    if xs is None:
        tr_xf, tr_spk = xf[:int(args.train_test*len(xf))], \
                        spk_ids[:int(args.train_test*len(spk_ids))]
        te_xf, te_spk = xf[int(args.train_test*len(xf)):], \
                        spk_ids[int(args.train_test*len(spk_ids)):]
        val_xf, val_spk = tr_xf[int(0.9*len(tr_xf)):], \
                            tr_spk[int(0.9*len(tr_spk)):]
        tr_xf, tr_spk = tr_xf[:int(0.9*len(tr_xf))], \
                        tr_spk[:int(0.9*len(tr_spk))]
    if xf is None:
        tr_xs, tr_spk = xs[:int(args.train_test*len(xs))], \
                        spk_ids[:int(args.train_test*len(spk_ids))]
        te_xs, te_spk = xs[int(args.train_test*len(xs)):], \
                        spk_ids[int(args.train_test*len(spk_ids)):]
        val_xs, val_spk = tr_xs[int(0.9*len(tr_xs)):], \
                            tr_spk[int(0.9*len(tr_spk)):]
        tr_xs, tr_spk = tr_xs[:int(0.9*len(tr_xs))], \
                        tr_spk[:int(0.9*len(tr_spk))]
    if xs is not None and xf is not None:
        tr_xf, tr_xs, tr_spk = xf[:int(args.train_test*len(xf))], \
                                xs[:int(args.train_test*len(xs))], \
                                spk_ids[:int(args.train_test*len(spk_ids))]
        te_xf, te_xs, te_spk = xf[int(args.train_test*len(xf)):], \
                                xs[int(args.train_test*len(xs)):], \
                                spk_ids[int(args.train_test*len(spk_ids)):]
        val_xf, val_xs, val_spk = tr_xf[int(0.9*len(tr_xf)):], \
                                    tr_xs[int(0.9*len(tr_xs)):], \
                                    tr_spk[int(0.9*len(tr_spk)):]
        tr_xf, tr_xs, tr_spk = tr_xf[:int(0.9*len(tr_xf))], \
                                tr_xs[:int(0.9*len(tr_xs))], \
                                tr_spk[:int(0.9*len(tr_spk))]


    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True
    }

    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': False,
        'drop_last': False
    }

    train_dataset = VidTimit(
        args, xf = tr_xf, xs = tr_xs, spk_ids = tr_spk
    )
    test_dataset = VidTimit(
        args, xf = te_xf, xs = te_xs, spk_ids = te_spk
    )
    val_dataset = VidTimit(
        args, xf = val_xf, xs = val_xs, spk_ids = val_spk
    )

    train_loader = DataLoader(train_dataset, **train_params)
    test_loader = DataLoader(test_dataset, **test_params)
    val_loader = DataLoader(val_dataset, **test_params)


    return train_loader, test_loader, val_loader