:: Run Test -Evaluate
cd ./src
::Initilaize IMG_SIZE
python ./__init__.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
cd ./test
python ./test.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)
python ./evaluate.py
IF %ERRORLEVEL% NEQ 0 (Echo An error was found &Exit /b 1)