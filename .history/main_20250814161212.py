import argparse
import torch
from config import get_default_config
from data_loader import DatasetLoader, mixed_collate
from models.snn import CombinedModel
from loss_function import CoxLoss
from train_test import train, test, evaluate_test_set
from util import plot_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNN survival model")
    parser.add_argument("--mode", type=str, default="omic", choices=["omic", "path", "pathomic"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    

    # Data
    train_loader = torch.utils.data.DataLoader(
        DatasetLoader(args.data_cv, 'train', args.mode),
        batch_size=args.batch_size, shuffle=True, collate_fn=mixed_collate
    )
    val_loader = torch.utils.data.DataLoader(
        DatasetLoader(cfg.data_cv, 'test', args.mode),
        batch_size=args.batch_size, shuffle=False, collate_fn=mixed_collate
    )

    # Model
    model = CombinedModel(cfg.input_size, cfg.output_size, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    train_losses, test_losses, c_indices, train_cindexs = [], [], [], []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        loss, train_cindex = train(train_loader, model, CoxLoss, optimizer, args.device)
        train_losses.append(loss)
        test_loss, c_index = test(val_loader, model, CoxLoss, args.device)
        test_losses.append(test_loss)
        c_indices.append(c_index)
        train_cindexs.append(train_cindex)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()

    # Save best model
    torch.save(best_model_state, "best_model.pth")

    # Final eval
    model.load_state_dict(best_model_state)
    evaluate_test_set(model, val_loader, CoxLoss, args.device)

    # Plot
    plot_metrics(train_losses, test_losses, train_cindexs, c_indices, args.epochs)
