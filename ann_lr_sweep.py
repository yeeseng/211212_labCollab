import subprocess

learning_rates = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1]

for lr in learning_rates:
    subprocess.run(['python', '130_basicModel03.py', '--lr', str(lr)])