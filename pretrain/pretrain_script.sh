

# '../experiment/500xCompressor' is your config path, examples is correct

python pre_prepare_data.py --work_dir '../experiment/500xCompressor'
python ./pre_trainer.py --work_dir '../experiment/500xCompressor' --port 14572
python ./pre_evaluator.py --work_dir '../experiment/500xCompressor' --batch_size 1