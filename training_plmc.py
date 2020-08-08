import argparse
from dataset import get_dataset
from models import petrained_language_model_classifier, train
import os
import shutil
import torch

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=['20_simplified', '6_simplified'], help="Dataset selected.")
    parser.add_argument("model_type", type=str, choices=['bert', 'distilbert'], help="Petrained natural language model type selected.")
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
    
    args = parser.parse_args()

    dataset2num_labels = {
        '20_simplified': 20,
        '6_simplified': 6,
    }

    models_use_token_tipe_ids = frozenset({
        "bert",
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    conf_name = f"{args.num_train_epochs}_{args.batch_size}_{args.max_length}_{args.learning_rate}_{args.num_warmup_steps}_{args.adam_epsilon}_{args.weight_decay}_{args.max_grad_norm}_{args.seed}"
    use_token_type_ids = True if args.model_type in models_use_token_tipe_ids else False
    num_classes = dataset2num_labels[args.dataset]
    
    model_tokenizer, model_classifier = petrained_language_model_classifier(
        args.model_type, 
        dataset2num_labels[args.dataset], 
        lowercase=args.lowercase, 
    )

    train_set, validation_set, label2id, _, _ = get_dataset(
        model_tokenizer, 
        args.dataset, 
        max_length=args.max_length, 
        seed=args.seed, 
        use_token_type_ids=use_token_type_ids,
    )
    
    model_path = os.path.join(args.output_dir, args.dataset, args.model_type, conf_name)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
            
    os.makedirs(model_path)

    id2label = { value:key for (key, value) in label2id.items()}
    
    train(model_classifier, train_set, validation_set, device, model_path, id2label, batch_size=args.batch_size, weight_decay=args.weight_decay, 
        num_train_epochs=args.num_train_epochs, lr=args.learning_rate, eps=args.adam_epsilon, num_warmup_steps=args.num_warmup_steps, 
        max_grad_norm=args.max_grad_norm, seed=args.seed, use_token_type_ids=use_token_type_ids)
    

if __name__ == "__main__": 
    main()  