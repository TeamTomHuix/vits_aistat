import os


if __name__ == "__main__":
    os.chdir('multirun/')
    for folder in os.listdir('.'):
        os.chdir(f'{folder}')
        for item in os.listdir('.'):
            os.chdir(f'{item}')
            for last_item in os.listdir('.'):
                if os.path.isdir(f'{last_item}'):
                    os.chdir(f'{last_item}')
                    print('Dossiers', last_item, os.listdir('.'))
                    print()
                    os.system(f'wandb sync --sync-all')
                    os.chdir('..')
            os.chdir('..')
        os.chdir('..')

