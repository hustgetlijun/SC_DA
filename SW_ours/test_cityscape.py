
import os

net = "vgg16"
part = "test_s"

start_epoch = 10   #9
max_epochs =  10    #9

output_dir = "./data/experiments/SW_Faster_ICR_CCR/cityscape/result"
dataset = "cityscape"

for i in range(start_epoch, max_epochs + 1):
    
    model_dir = "~/GLJ/CR-DA-DET/SW_Faster_ICR_CCR/reslut/experiments/SW_Faster_ICR_CCR/cityscape/ICR_CCR_23712/cityscape_{}.pth".format(
        i
    )
    command = "python eval/test_SW_ICR_CCR.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    )
    os.system(command)
