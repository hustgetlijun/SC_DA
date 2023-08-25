import os

net = "vgg16"
part = "test_s"
start_epoch = 7
max_epochs = 7
output_dir = "../data/experiments/SW_Faster_ICR_CCR/cityscape/result"
dataset = "cityscape"

for i in range(start_epoch, max_epochs + 1):
    model_dir = " ~/GLJ/CR-DA-DET/SW_Faster_ICR_CCR/data/experiments/Faster_ori/cityscape/cityscape_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=1 python eval/test.py --cuda --part {} --net {} --dataset {} --model_dir {}".format(
        part, net, dataset, model_dir
    )
    os.system(command)
