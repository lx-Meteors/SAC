# '../experiment/test_time' is your config path, examples is correct
python instruction_prepare_data.py --work_dir  '../experiment/test_time'
python ./instruction_trainer.py --work_dir   '../experiment/test_time' --port 14527 > train.log 2>&1 &
python ./instruction_evaluator.py --work_dir   '../experiment/test_time' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/test_time'
python ../util/evaluate_iid.py --work_dir '../experiment/test_time'