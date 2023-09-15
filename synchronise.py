import os


if __name__ == "__main__":
    for folder in os.listdir('multirun/'):
        os.system(f'cd {folder}')
        os.system(f'wandb sync --sync-all')
        os.system(f'cd ..')