# '../experiment/test_time' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/test_time'
python ./pre_trainer.py --work_dir '../experiment/test_time' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/test_time' --batch_size 1