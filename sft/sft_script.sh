
CUDA_VISIBLE_DEVICES=0 python ./instruction_evaluator.py --work_dir   '../experiment/500x_time' --batch_size 1



python ../util/evaluate_ood.py --work_dir  '../experiment/main_experiment/500x_DPL_1B_MultiChunk'
python ../util/evaluate_iid.py --work_dir  '../experiment/main_experiment/500x_DPL_1B_MultiChunk'
