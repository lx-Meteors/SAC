


# '../experiment/500xCompressor' is your config path, examples is correct


python instruction_prepare_data.py --work_dir  '../experiment/500xCompressor'
python ./instruction_trainer.py --work_dir   '../experiment/500xCompressor' --port 14527 > train.log 2>&1 &
python ./instruction_evaluator.py --work_dir   '../experiment/500xCompressor' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/500xCompressor'
python ../util/evaluate_iid.py --work_dir '../experiment/500xCompressor'