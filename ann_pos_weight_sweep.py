import subprocess

positive_weights = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

for pos_weight in positive_weights:
    subprocess.run(['python', '130_basicModel03.py', '--pos_weight', str(pos_weight)])