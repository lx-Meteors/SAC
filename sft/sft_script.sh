


python ./instruction_evaluator.py --work_dir   '../experiment/sac_time' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/sac_time'
python ../util/evaluate_iid.py --work_dir '../experiment/sac_time'
