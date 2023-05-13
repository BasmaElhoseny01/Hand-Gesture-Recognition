$train_path = ".\training_pipeline"
$test_path = ".\test_pipeline"
cd $train_path
py .\main.py
cd ..
cd $test_path
py .\main.py
py .\evaluate.py
cd ..