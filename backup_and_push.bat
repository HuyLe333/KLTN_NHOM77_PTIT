@echo off
chcp 65001 > nul
echo =======================================================
echo   BAT DAU QUA TRINH SAO LUU DATABASE VA PUSH LEN GIT
echo =======================================================

rem 1. Chay file python de xuat database ra file CSV va Schema SQL
echo.
echo [1/3] Dang chay script Python de export du lieu tu MySQL...
python export_database.py

rem 2. Thuc hien cac lenh Git
echo.
echo [2/3] Dang them cac file thay doi vao Git...
git add .

echo.
echo Danh sach cac file se duoc commit:
git status -s

echo.
set "commit_msg=Cap nhat ma nguon va Schema database moi nhat"
set /p commit_msg="Nhap Commit Message (nhan Enter de dung mac dinh: '%commit_msg%'): "

echo.
echo [3/3] Dang thuc hien commit...
git commit -m "%commit_msg%"

echo.
echo Dang push du lieu len GitHub...
git push

if errorlevel 1 (
    echo.
    echo [LOI] Day code len GitHub that bai.
) else (
    echo.
    echo =======================================================
    echo   THANH CONG! Du lieu database va ma nguon da duoc cap nhat.
    echo =======================================================
)

pause
