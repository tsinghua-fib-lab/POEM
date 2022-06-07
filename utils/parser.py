import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="FM")


    parser.add_argument("--epoch", type=int, default=2, help="number of epochs")
    parser.add_argument("--train_size", type=int, default=50000, help="batch size")
    parser.add_argument("--test_size", type=int, default=50000, help="batch size")
    parser.add_argument("--dim", type=int, default=8, help="embedding size")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="l2 regularization weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=3, help="gpu id")
    parser.add_argument("--model", type=str, default='dfm', help="model_type")
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument("--node_dropout", type=bool, default=False, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--mlp_dims", type=int, default=(64,64), help="ratio of node dropout")
    parser.add_argument("--dfm_dropout", type=float, default=0.2, help="ratio of node dropout")
    parser.add_argument("--dfm_emb_dim", type=int, default=8, help="ratio of node dropout")
    parser.add_argument("--dataset", type=str, default='data_a', help="ratio of node dropout")
    parser.add_argument("--atten_emb_dim", type=int, default=64, help="ratio of node dropout")
    parser.add_argument("--afi_mlp_dims", type=int, default=(400,400), help="ratio of node dropout")
    parser.add_argument("--afi_num_heads", type=int, default=2, help="ratio of node dropout")
    parser.add_argument("--afi_num_layers", type=int, default=3, help="ratio of node dropout")
    parser.add_argument("--xdfm_cross_layer_sizes", type=int, default=(16,16), help="ratio of node dropout")
    parser.add_argument("--agg", type=str, default='mixed', help="type of aggregator")
    parser.add_argument("--sensi_mlp_dims", type=int, default=(64, 64), help="type of aggregator")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    return parser.parse_args()