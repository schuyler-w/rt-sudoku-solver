# rt-sudoku-solver
Real Time Webcam Sudoku Solver

Solves sudoku in real time using your computer's webcam feed

This program uses computer vision (CV) techniques to extract the sudoku from a live camera feed and displays the solution onto the frame after solving the puzzle, close to an augmented reality solver. It also crops out the latest solved sudoku image and saves it as "solvedSudoku.jpg".

main.py uses Peter Norvig's algorithm using constraint propagation and backtracking search to solve sudoku problems in under 0.01 seconds (https://norvig.com/sudoku.html)

Digit recognizer trained using Char74k Dataset and by centralizing according to center of mass achieved an accuracy of 99.5% within 5 epochs. 
