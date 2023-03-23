class UAIConfig(object):
    def __init__(self, args):
        super(UAIConfig, self).__init__()
        self.x_shape = (512,)
        self.spks = args.spks
        self.dropout = args.dropout
        self.embed_dim1 = args.emb_dim_1
        self.embed_dim2 = args.emb_dim_2