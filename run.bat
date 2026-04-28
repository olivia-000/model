@REM Description: Batch file to start the firetruck detection on windows startup
@REM Move this file to the startup folder of windows (C:\Users\%username%\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup)
@REM Windows' short-cut to open that folder: press windows+R, then type `shell:startup` 

@REM Activate the Conda environment with the required dependencies
Call conda activate firetruck-detection

@REM Run the main.py file
cd firetruck_detection
python main.py
pause