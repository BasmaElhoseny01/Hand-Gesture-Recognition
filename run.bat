:: %1 Preprocessing option
:: %2 feature extractor option
:: %3 model Option

:: Remove Previous Result to prevent wrong Accuracy if model Failed
cd ./results
del *.txt
cd ..


::Remove Old Models
cd ./models
del *.joblib
cd ..

:: Run Train - Test -Evaluate
cd ./src
python ./train.py %1 %2 %3
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
python ./test.py %1 %2 %3
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
python ./evaluate.py %1 %2 %3
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)


