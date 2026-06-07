import argparse

from train import train
from predict import predict


def main():
    parser = argparse.ArgumentParser(description='ZSL ResNet — Zero-Shot Animal Classification')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_p = subparsers.add_parser('train')
    train_p.add_argument('--dataset_path',   required=True)
    train_p.add_argument('--num_epochs',     type=int,   default=10)
    train_p.add_argument('--eval_interval',  type=int,   default=100)
    train_p.add_argument('--learning_rate',  type=float, default=0.00005)
    train_p.add_argument('--output_filename',type=str,   default='model.pth')
    train_p.add_argument('--batch_size',     type=int,   default=32)

    pred_p = subparsers.add_parser('predict')
    pred_p.add_argument('--dataset_path', required=True)
    pred_p.add_argument('--model_path',   type=str, default='model.pth')
    pred_p.add_argument('--output_csv',   type=str, default='zsl_predictions.csv')

    args = parser.parse_args()

    if args.command == 'train':
        train(
            dataset_path    = args.dataset_path,
            num_epochs      = args.num_epochs,
            eval_interval   = args.eval_interval,
            learning_rate   = args.learning_rate,
            output_filename = args.output_filename,
            batch_size      = args.batch_size
        )

    elif args.command == 'predict':
        predict(
            dataset_path = args.dataset_path,
            model_path   = args.model_path,
            output_csv   = args.output_csv
        )


if __name__ == '__main__':
    main()