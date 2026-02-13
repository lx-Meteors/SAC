


CUDA_VISIBLE_DEVICES=1 python ./instruction_evaluator.py --work_dir   '../experiment/sac_time' --batch_size 10
python ../util/evaluate_ood.py --work_dir '../experiment/sac_time'
python ../util/evaluate_iid.py --work_dir '../experiment/sac_time'
