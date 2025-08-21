import argparse
import torch
import pickle
import pandas as pd
from data_loader import DatasetLoader, mixed_collate
from loss_function import CoxLoss
from train_test import train, test, evaluate_test_set
from util import plot_metrics
from network import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNN survival model")
    parser.add_argument("--mode", type=str, default="omic", choices=["omic", "path", "pathomic"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="Train survival model")

    # Data settings
    parser.add_argument("--root", type=str, default="data/TCGA_GBMLGG/",
                        help="資料根目錄")
    parser.add_argument("--split_file", type=str,
                        default="data/TCGA_GBMLGG/splits/0506_single_split_all_st.pkl",
                        help="資料路徑")
    parser.add_argument("--mode", type=str, default="omic", choices=["omic", "path", "pathomic"],
                        help="模型模式")

    # Model settings
    parser.add_argument("--model", type=str, default="snn", choices=["snn", "psnn", "cnn","fused","pfused"],
                        help="模型名稱")
    parser.add_argument("--input_size", type=int, default=4088, help="模型輸入維度")
    parser.add_argument("--output_size", type=int, default=1, help="模型輸出維度")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout 機率")
    parser.add_argument("--dim_1", type=int, default=16, help="GatedMultimodalUnit 第一模態維度")
parser.add_argument("--dim_2", type=int, default=16, help="GatedMultimodalUnit 第二模態維度")
parser.add_argument("--output_dim", type=int, default=8, help="GatedMultimodalUnit 輸出維度")

    # Training settings
    parser.add_argument("--epochs", type=int, default=50, help="訓練 epoch 數")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="運行裝置")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    # Load pathway mask
    args.pathway_mask = torch.tensor(
        pd.read_csv(args.root + "pathway_mask.csv", index_col=0).values,
        dtype=torch.float32
    )

    # Load data_cv
    args.data_cv = pickle.load(open(args.split_file, 'rb'))
    args.data_cv.pop("data_pd")

    return args


if __name__ == "__main__":
    args = parse_args()
    

    # Data
    train_loader = torch.utils.data.DataLoader(
        DatasetLoader(args.data_cv, 'train', args.mode),
        batch_size=args.batch_size, shuffle=True, collate_fn=mixed_collate
    )
    val_loader = torch.utils.data.DataLoader(
        DatasetLoader(args.data_cv, 'test', args.mode),
        batch_size=args.batch_size, shuffle=False, collate_fn=mixed_collate
    )

    # Model
    # model = CombinedModel(args.input_size, args.output_size, args.dropout).to(args.device)
   # 待改
    model = get_model(args.model,
                      input_size=args.input_size,
                      output_size=args.output_size,
                      dropout_rate=args.dropout).to(args.device)
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
