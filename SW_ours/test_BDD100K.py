import os

net = "vgg16"
part = "test_t"
start_epoch = 6
max_epochs =  12

output_dir = "./data/experiments/SW_Faster_ICR_CCR/BDD100K/result"
dataset = "BDD100K"

for i in range(start_epoch, max_epochs + 1):
    model_dir = "./data/experiments/SW_Faster_ICR_CCR/BDD100K/ours/BDD100K_{}.pth".format(
        i
    )
    command = "CUDA_VISIBLE_DEVICES=2 python eval/test_SW_ICR_CCR_change_test.py --cuda  --lc --gc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i
    ) #model_allover_128c_1_no_instance_DA_three_change_weight_two_context_1x1_cs_BCE_init_0
    # command = "CUDA_VISIBLE_DEVICES=1 python eval/test_SW_ICR_CCR_change.py --cuda  --lc --gc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
    #     part, net, dataset, model_dir, output_dir, i
    # )#model_allover_128c_1_no_instance_three_change_weight_3

    os.system(command)

    # command = "python eval/test_SW_pix.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
    #     part, net, dataset, model_dir, output_dir, i
    # )
