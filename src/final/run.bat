:: %1 Preprocessing option
:: %2 feature extractor option
:: %3 model Option

:: Run Test -Evaluate
cd ./src
::Initilaize IMG_SIZE
python ./__init__.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
cd ./final
python ./test.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
python ./evaluate.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)