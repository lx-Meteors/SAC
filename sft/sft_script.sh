# '../experiment/main_experiment_5x' is your config path, examples is correct
python instruction_prepare_data.py --work_dir  '../experiment/main_experiment_5x'
python ./instruction_trainer.py --work_dir   '../experiment/main_experiment_5x' --port 14527 > train.log 2>&1 &
python ./instruction_evaluator.py --work_dir   '../experiment/main_experiment_5x' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/main_experiment_5x'
python ../util/evaluate_iid.py --work_dir '../experiment/main_experiment_5x'