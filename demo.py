from main import parse, train
from src.model import FUSIONNET

if __name__ == "__main__":
	args = parse(print_help=True)
	# args.split_mode = 'local'
	# args.n_splits = -1
	args.dataset_name = 'Cdataset'
	# args.lr = 1e-3
	args.lr = 5e-4
	# # args.use_bn = True
	args.dropout = 0.4
	# 0 -1(all)
	args.disease_neighbor_num = 7
	args.drug_neighbor_num = 7
	args.disease_feature_topk = 3000
	args.drug_feature_topk = 3000
	args.embedding_dim = 128
	args.neighbor_embedding_dim = 64
	# args.hidden_dims = (64, 32)
	# args.debug = True
	args.epochs = 33
	args.train_fill_unknown = False
	args.use_interaction = True

	args.use_linear = True
	# args.comment = "test"
	args.loss_fn = "focal"

	# args.alpha = 0.8
	args.lamda = 1

	train(args, FUSIONNET)
