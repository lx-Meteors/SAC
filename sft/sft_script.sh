


# '../experiment/ICAE' is your config path, examples is correct
python instruction_prepare_data.py --work_dir  '../experiment/ICAE'
python ./instruction_trainer.py --work_dir   '../experiment/ICAE' --port 14527 > train.log 2>&1 &
python ./instruction_evaluator.py --work_dir   '../experiment/ICAE' --batch_size 1
python ../util/evaluate_ood.py --work_dir '../experiment/ICAE'
python ../util/evaluate_iid.py --work_dir '../experiment/ICAE'