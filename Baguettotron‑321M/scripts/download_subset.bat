@echo off
REM Quick script to download SYNTH dataset subsets for testing (Windows)

echo ========================================
echo SYNTH Dataset Subset Downloader
echo ========================================
echo.

REM Default values
set SAMPLES=%1
if "%SAMPLES%"=="" set SAMPLES=100

set OUTPUT_DIR=%2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=data\demo

echo Configuration:
echo   Max samples: %SAMPLES%
echo   Output directory: %OUTPUT_DIR%
echo.

REM Check if we're in the right directory
if not exist prepare_synth_data.py (
    echo Warning: prepare_synth_data.py not found
    echo Please run this script from the Baguettotron-321M root directory
    exit /b 1
)

REM Run the download
echo Downloading %SAMPLES% samples...
python prepare_synth_data.py --subset --max-samples %SAMPLES% --tokenize --output-dir %OUTPUT_DIR%

echo.
echo Done! Dataset ready at: %OUTPUT_DIR%
echo.
echo Quick commands:
echo   # View the data
echo   dir %OUTPUT_DIR%
echo.
echo   # Start training with this subset
echo   python train.py --train-data %OUTPUT_DIR%\train_tokens.json --max-epochs 1
