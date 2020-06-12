import argparse
from dataset import load_20newsgroup_segments, generate_dataset_20newsgroup_segments
from models import RoBert, train
import os
import shutil
import torch

from collections import Counter, defaultdict

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['20_simplified', '6_simplified'], help="Dataset selected.")
    parser.add_argument("pretrained_model", type=str, default="bert-base-uncased", help="Petrained natural language model selected.")
    parser.add_argument("--output_dir", type=str, default="./saved_models", help="The output directory where the results and checkpoints will be written.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128, help="The maximum total input sequence length after tokenization.") # check for each dataset
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lowercase", action='store_true', help="Uses an uncased model.")
    parser.add_argument("--lstm_hidden_dim", type=int, default=1024, help="Number of LSTM units.")
    parser.add_argument("--dense_hidden_dim", type=int, default=32, help="Number oh dense units.")
    parser.add_argument("--num_lstm_layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--num_dense_layers", type=int, default=2, help="Number of dense layers.")
    parser.add_argument("--size_segment", type=int, default=200, help="Size segment.")
    parser.add_argument("--size_shift", type=int, default=50, help="Size shift.")
    parser.add_argument("--lr_lstm", type=float, default=0.001, help="Learning rate of LSTM.")
    parser.add_argument("--epochs_decrease_lr_lstm", type=int, default=3, help="Number of epoch to descrease learning rate when the validation loss does not improve.")
    parser.add_argument("--reduced_factor_lstm", type=float, default=0.95, help="Reduced factor for learning rate of the LSTM.")

    args = parser.parse_args()

    dataset2num_labels = {
        '20_simplified': 20,
        '6_simplified': 6,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_token_type_ids = True

    conf_name = (f"{args.num_train_epochs}_{args.batch_size}_{args.max_length}_{args.learning_rate}_"
      f"{args.num_warmup_steps}_{args.adam_epsilon}_{args.weight_decay}_{args.max_grad_norm}_{args.seed}_"
      f"{args.lstm_hidden_dim}_{args.dense_hidden_dim}_{args.num_lstm_layers}_{args.num_lstm_layers}"
      f"{args.lr_lstm}_{args.epochs_decrease_lr_lstm}_{args.reduced_factor_lstm}"
    )

    num_classes = dataset2num_labels[args.dataset]

    model_classifier = RoBert(
        num_classes, 
        bert_pretrained=args.pretrained_model, 
        device=device, 
        lstm_hidden_dim=args.lstm_hidden_dim, 
        dense_hidden_dim=args.dense_hidden_dim, 
        num_lstm_layers=args.num_lstm_layers,
        num_dense_layers=args.num_dense_layers, 
    )

    model_tokenizer = model_classifier.tokenizer()

    X_train, y_train, num_segments_train, X_test, y_test, num_segments_test, label2id = load_20newsgroup_segments(
        args.dataset, max_length=args.max_length, size_segment=args.size_segment, size_shift=args.size_shift)
        
    train_set = generate_dataset_20newsgroup_segments(X_train, y_train, num_segments_train, model_tokenizer, max_length=args.size_segment)
    validation_set = generate_dataset_20newsgroup_segments(X_test, y_test, num_segments_test, model_tokenizer, max_length=args.size_segment)
    
    model_path = os.path.join(args.output_dir, args.dataset, args.pretrained_model, conf_name)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
            
    os.makedirs(model_path)

    id2label = { value:key for (key, value) in label2id.items()}
    
    train(model_classifier, train_set, validation_set, device, model_path, id2label, batch_size=args.batch_size, weight_decay=args.weight_decay, 
        num_train_epochs=args.num_train_epochs, lr=args.learning_rate, eps=args.adam_epsilon, num_warmup_steps=args.num_warmup_steps, 
        max_grad_norm=args.max_grad_norm, seed=args.seed, use_token_type_ids=use_token_type_ids, lr_lstm=args.lr_lstm, epochs_decrease_lr_lstm=args.epochs_decrease_lr_lstm, 
        reduced_factor_lstm=args.reduced_factor_lstm)
    

if __name__ == "__main__": 
    main()  