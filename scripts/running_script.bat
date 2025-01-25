@echo off

rem Define the program executable path
set program=python3.8

rem Loop through each triple of arguments
for %%a in (12 15 18 21 24 27 30) do (
    for /l %%i in (1,1,10) do (
        rem Run the program with the current set of arguments
        echo Running %program% ./matching-algorithm-FL.py with arguments: --a 0 --n %%a --k 3 Run %%i
        %program% ./matching-algorithm-FL.py --a 0 --n %%a --k 3 --tc 4
    )
)

echo All runs completed.