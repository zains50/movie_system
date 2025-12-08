import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("recall_results.csv")

recalls = [20,40,80,100,300]
epochs = [x for x in range(0, 238+1, 2)]

test_recall_str = "test_k@"
train_recall_str = "train_k@"

train_loss = "train_loss"
val_loss = "val_loss"

train_recall_df = df["test_k@300"]
print(train_recall_df)

def make_recall_graph(epochs,recalls):

    for r in recalls:
        recall_nums = df[f"test_k@{r}"]
        x = epochs
        plt.plot(x,recall_nums,label=f"Recall@{r}")

    plt.title("Recall@K Over Training Epochs: Test", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Recall", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def make_loss_graph(epochs):
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]

    plt.plot(epochs,train_loss,label="Train Loss")
    plt.plot(epochs,val_loss,label="Test Loss")


    plt.title("Train and Test Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Recall", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

#
# make_recall_graph(epochs,recalls)
make_loss_graph(epochs)