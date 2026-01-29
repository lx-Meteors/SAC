# '../experiment/Chunck-Last-Token' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/Chunck-Last-Token'
python ./pre_trainer.py --work_dir '../experiment/Chunck-Last-Token' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/Chunck-Last-Token' --batch_size 1