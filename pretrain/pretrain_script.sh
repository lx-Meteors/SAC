# '../experiment/test_experiment' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/test_experiment'
python ./pre_trainer.py --work_dir '../experiment/test_experiment' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/test_experiment' --batch_size 1