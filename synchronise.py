import os


if __name__ == "__main__":
    for folder in os.listdir('multirun/'):
        os.system(f'cd {folder}')
        for item in os.listdir(f'multirun/{folder}'):
            os.system(f'cd {item}')
            os.system(f'wandb sync --sync-all')
            os.system(f'cd ..')
        os.system(f'cd ..')