:: %1 Preprocessing option
:: %2 feature extractor option
:: %3 model Option

:: Remove Previous Result to prevent wrong Accuracy if model Failed
cd ./results
del *.txt
cd ..


::Remove Old Models
@REM cd ./models
@REM del *.joblib
@REM cd ..

:: Run Train - Test -Evaluate
cd ./src
::Initilaize IMG_SIZE
python ./__init__.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
@REM python ./train.py %1 %2 %3
@REM IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
python ./test.py %1 %2 %3
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
python ./evaluate.py %1 %2 %3
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)