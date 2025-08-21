import matplotlib.pyplot as plt

def plot_metrics(train_losses, test_losses, train_cindexs, c_indices, epochs, model):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    axs[0].plot(range(epochs), train_losses, label='Train Loss', color='blue')
    axs[0].plot(range(epochs), test_losses, label='Validation Loss', color='orange')
    axs[0].set_title('Loss over epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(epochs), c_indices, label='Validation C-index', color='red')
    axs[1].plot(range(epochs), train_cindexs, label='Train C-index', color='green')
    axs[1].set_title('C-index over epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('C-index')
    axs[1].legend()
    axs[1].grid(True)
    fig.suptitle(f"{epochs} epochs on {model.}", fontsize=16)
    plt.tight_layout()
    plt.show()
