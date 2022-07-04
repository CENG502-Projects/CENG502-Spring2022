import matplotlib.pyplot as plt
import os
import time


def plot_result(
    max_epoch,  # max_epoch
    model_name,
    trainingLosses,
    validationLosses,
    total_time,
    learning_rate,
    optimizer_name,
    details,
):
    classes = [1, 2, 3, 4, 5]
    epoch_number = []
    for i in range(len(trainingLosses)):
        epoch_number.append(i + 1)

    fig, (
        ax0,
        ax2,
    ) = plt.subplots(2, figsize=(10, 14))

    plt.subplots_adjust(hspace=0.3, wspace=0.8, left=0.08, right=0.92, top=0.9)
    fig.tight_layout(pad=3.0)

    fig.suptitle(
        model_name
        + "\nNumber of epochs: "
        + str(max_epoch)
        + "\nLast done epoch: "
        + str(len(trainingLosses))
        + "\nTotal time: "
        + str(total_time // 3600)
        + ":"
        + str((total_time % 3600) // 60)
        + ":"
        + str(total_time % 60)
        + "\nLearning rate: "
        + str(learning_rate)
        + "\nOptimizer name: "
        + optimizer_name
        + "\nDetails: "
        + details,
        fontsize=12,
    )

    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.axis("off")

    ax2.plot(epoch_number, trainingLosses, ".-")
    ax2.plot(epoch_number, validationLosses, ".-")
    ax2.set_title("Loss on every epoch", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.set_xticks(range(1, len(trainingLosses) + 1))
    ax2.tick_params("x", labelrotation=90)
    for i, j in zip(epoch_number, trainingLosses):
        ax2.annotate(str("{:.2f}".format(j)), xy=(i, j), fontsize=6)
    for i, j in zip(epoch_number, validationLosses):
        ax2.annotate(str("{:.2f}".format(j)), xy=(i, j), fontsize=6)
    ax2.legend(["Training", "Validation"])

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists("./figures"):
        os.mkdir("./figures")
    name_plot = "figures/" + model_name + "_" + timestr + ".svg"
    plt.savefig(name_plot, orientation="portrait", format="svg")
