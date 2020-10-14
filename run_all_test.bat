@echo off

rem cnn
python run_kfold_test.py --network cnn --task home
python eval_kfold_test.py --network cnn --task home

rem cnn-resnet
python run_kfold_test.py --network cnn-resnet --task home
python eval_kfold_test.py --network cnn-resnet --task home

rem cnn-cbam
python eval_kfold_test.py --network cnn-cbam --task home
python run_kfold_test.py --network cnn-cbam --task home