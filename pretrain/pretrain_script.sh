# '../experiment/Random' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/Random'
python ./pre_trainer.py --work_dir '../experiment/Random' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/Random' --batch_size 1