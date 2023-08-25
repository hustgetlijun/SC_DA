import os

net = "res101"
part = "test_t"
start_epoch = 9
max_epochs = 12
output_dir = "./data/experiments/SW_Faster_ICR_CCR/water/model"
dataset = "water"

for i in range(start_epoch, max_epochs + 1):
    model_dir = "./data/experiments/SW_Faster_ICR_CCR/clipart/model_result/model_allover_all/ours/water_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=2 python eval/test_SW_ICR_CCR_change_test.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
