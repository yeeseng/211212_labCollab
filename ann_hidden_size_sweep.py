import subprocess

hidden_sizes = [4, 8, 16, 32, 64, 128, 256]

for hidden_size in hidden_sizes:
    subprocess.run(['python', '130_basicModel03.py', '--hidden_size', str(hidden_size)])