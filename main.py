from src.train import train
from src.test import test
from src.visualize import visualize_sequence, plot_losses

if __name__ == "__main__":
    print("Starting main execution")
    data_root = "data"
    print(f"Training with data from {data_root}")
    train(data_root)
    print("Training completed, starting testing")
    test(data_root)
    print("Testing completed, starting visualization")
    visualize_sequence("data/val/dancetrack0004", "results/val_results.txt")
    print("Visualization completed, plotting losses")
    plot_losses()
    print("Main execution completed")
