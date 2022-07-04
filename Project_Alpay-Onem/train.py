import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from math import log10, sqrt
import matplot_helper as mph
import dataset
import model
import time
import copy
import wandb
import click

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def saveImagesToDisk(H_estimateds, H_estimateds_tm, epochNo, iterNo):
    folder = os.path.join("./valid_images", "epoch" + str(epochNo))
    if not os.path.exists(folder):
        os.makedirs(folder)
    H_estimateds_file = os.path.join(folder, "H_estimateds_no" + str(iterNo) + ".exr")
    H_estimateds_tm_file = os.path.join(
        folder, "H_estimateds_tm_no" + str(iterNo) + ".png"
    )
    cv2.imwrite(H_estimateds_file, H_estimateds)
    cv2.imwrite(H_estimateds_tm_file, H_estimateds_tm)


def PSNR(ours, original, max_pixel):
    mse = np.mean((original - ours) ** 2)

    if mse == 0:
        return 100
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def calculatePSNRs(H_estimateds, gts, H_estimateds_tm, H_gts_tm):
    psnr_l = PSNR(H_estimateds, gts, 1.0)  #! what should max pixel be?
    psnr_mu = PSNR(H_estimateds_tm, H_gts_tm, 255.0)  #! what should max pixel be?

    return psnr_l, psnr_mu


def train(
    epoch_count,
    dataPath,
    useCuda,
    batch_size,
    validation_path,
    no_wandb,
    model_name,
    details,
):
    def wandb_log(*args, **kwargs):
        if no_wandb:
            return
        wandb.log(*args, **kwargs)

    myNetwork = model.MCANet()

    if useCuda:
        print("Using cuda")
        myNetwork.cuda()

    start_time = time.time()

    transforms = dataset.getTransforms()
    dataLoaders = dataset.getDataLoaders(
        transforms, batch_size, dataPath, validation_path
    )

    params_to_update = myNetwork.parameters()
    learningRate = 0.00001
    criterion = nn.L1Loss()
    optimizer = optim.Adam(params_to_update, lr=learningRate)

    losses_epochwise = {"train": [], "valid": []}

    print("Starting Training")

    # to_save_hacky = []
    # val_images = np.load("val_image_18x256x256.npy", allow_pickle=True)

    wandb_step = 0

    for epoch in range(int(epoch_count)):
        # for phase in ["train", "valid"]:
        for phase in ["train"]:
            if phase == "train":
                myNetwork.train()
            else:
                myNetwork.eval()

            torch.cuda.empty_cache()

            running_loss = 0.0

            for i, data in enumerate(dataLoaders[phase]):
                if phase == "train":
                    wandb_step += 1

                inputs, gts = data

                ############
                # if epoch < 5:
                #     # img = inputs[0]
                #     # img = img.transpose((2, 0, 1))
                #     to_save_hacky.append(data)
                #     continue
                # else:
                #     np.save("val_image_18x256x256", to_save_hacky)
                #     exit()
                ############

                if useCuda:
                    inputs = inputs.cuda()
                    gts = gts.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = myNetwork(inputs)

                    H_estimateds = outputs
                    H_estimateds_tm = torch.log(1 + 5000 * H_estimateds) / np.log(
                        1 + 5000
                    )
                    H_gts_tm = torch.log(1 + 5000 * gts) / np.log(1 + 5000)

                    loss = criterion(H_estimateds_tm, H_gts_tm)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    else:  # validation
                        if (epoch + 1) % 25 == 0:
                            hdrEst = (
                                H_estimateds.detach()
                                .cpu()
                                .numpy()[0]
                                .transpose((1, 2, 0))
                            )
                            hdrGt = gts.detach().cpu().numpy()[0].transpose((1, 2, 0))
                            ldrTmEst = (
                                H_estimateds_tm.detach()
                                .cpu()
                                .numpy()[0]
                                .transpose((1, 2, 0))
                                * 255.0
                            )
                            ldrTmGt = (
                                H_gts_tm.detach().cpu().numpy()[0].transpose((1, 2, 0))
                                * 255.0
                            )

                            psnr_l, psnr_mu = calculatePSNRs(
                                hdrEst, hdrGt, ldrTmEst, ldrTmGt
                            )

                            saveImagesToDisk(hdrEst, ldrTmEst, epoch + 1, i + 1)
                            print(
                                "Validation Scene No: ",
                                str(i + 1),
                                " PSNR_L: ",
                                str(psnr_l),
                                " PSNR_mu: ",
                                str(psnr_mu),
                            )
                            wandb_log(
                                {
                                    f"val/psnr_l/scene_{i+1}": psnr_l,
                                    f"val/psnr_mu/scene_{i+1}": psnr_mu,
                                },
                                step=wandb_step,
                            )

                running_loss += loss.item() * inputs.size(0)

                if phase == "train":
                    if (i + 1) % 2 == 0:
                        wandb_log({"loss/loss": loss.item()}, step=wandb_step)
                        print(
                            "Epoch [%d/%d], Batch [%d/%d], Loss: %.4f"
                            % (
                                epoch + 1,
                                int(epoch_count),
                                i + 1,
                                len(dataLoaders[phase].dataset) // batch_size,
                                loss.item(),
                            )
                        )

            epoch_loss = running_loss / len(dataLoaders[phase].dataset)

            print("Done phase: " + phase)
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, epoch_loss))
            if phase == "train":
                wandb_log({"loss/epoch_loss": epoch_loss}, step=wandb_step)
            losses_epochwise[phase].append(epoch_loss)
            losses_epochwise["valid"].append(0.0)

            if (epoch + 1) % 25 == 0:
                print(
                    "Saving model params for each 25th epoch. Current epoch: "
                    + str(epoch + 1)
                )
                modelParams = copy.deepcopy(myNetwork.state_dict())
                if not os.path.exists("./checkpoints"):
                    os.mkdir("./checkpoints")
                checkpointPath = "./checkpoints/checkpoint_" + str(epoch + 1) + ".pth"
                torch.save(modelParams, checkpointPath)

                ############
                # plot pre-selected random crops from test set
                # myNetwork.eval()
                #
                # img_stack = torch.tensor([])
                # for (val_input, val_gt) in val_images[0:1]:
                #     val_input_c = val_input.cuda()
                #     val_out = myNetwork(val_input_c)
                #     val_out = val_out.cpu()
                #     img_stack = torch.cat((img_stack,val_input.view(-1, 3, 256, 256)[[0,2,4],...], val_gt, val_out),dim=0)
                #
                # val_input_c = None
                # grid = torchvision.utils.make_grid(img_stack, nrow=5, padding=0)
                # wandb_log({"val_imgs": wandb.Image(grid)}, step=wandb_step)
                #
                # myNetwork.train()
                ############

        if ((epoch + 1) == 1) or ((epoch + 1) % 5 == 0):
            time_elapsed = time.time() - start_time
            # plot results until this epoch
            mph.plot_result(
                int(epoch_count),  # max_epoch
                model_name,
                losses_epochwise["train"],
                losses_epochwise["valid"],
                time_elapsed,
                0.00001,
                "Adam",
                details,
            )

    finish_time = time.time()
    time_elapsed = finish_time - start_time
    print(
        "Finished training in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return losses_epochwise, time_elapsed


@click.command()
@click.option("--epoch_count", type=int, default=50)
@click.option("--batch_size", type=int, default=5)
@click.option("--model_name", type=str, default="MCANet")
@click.option("--wandb_experiment_name", type=str, default="")
@click.option("--no_wandb", is_flag=True)
def main(epoch_count, batch_size, model_name, wandb_experiment_name, no_wandb):
    torch.manual_seed(43)  # Fix the seed for reproducible results

    useCuda = True

    data_path = "./dataset/SIGGRAPH17_HDR_Trainingset/Training"
    validation_path = "./dataset/SIGGRAPH17_HDR_Testset/Test"

    details = "Batch Size = " + str(batch_size)

    wandb_args = {
        "project": "mcanet",
        "entity": "kadircenk-cemonem-mcanet",
        "config": {
            "epoch_count": epoch_count,
            "batch_size": batch_size,
            "model_name": model_name,
        },
    }

    if wandb_experiment_name:
        wandb_args["name"] = wandb_experiment_name

    if not no_wandb:
        wandb.init(**wandb_args)

    losses, time_elapsed = train(
        epoch_count,
        data_path,
        useCuda,
        batch_size,
        validation_path,
        no_wandb,
        model_name,
        details,
    )


if __name__ == "__main__":
    main()
