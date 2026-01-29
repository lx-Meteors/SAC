# '../experiment/ICAE' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/ICAE'
python ./pre_trainer.py --work_dir '../experiment/ICAE' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/ICAE' --batch_size 1