import subprocess

Layers = [4, 8, 16, 32, 64, 128, 256]

for num_layers in Layers:
    subprocess.run(['python', '130_basicModel03.py', '--num_layers', str(num_layers)])