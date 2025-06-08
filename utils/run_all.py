import argparse
import os
import subprocess
import yaml
import time


def run_experiment(gpu_number='0,', target_file='run.yaml'):
    # read yaml file
    with open(target_file, 'r') as f:
        config = yaml.safe_load(f)

    for gpu in gpu_number.split(','):
        target_key = 'NVIDIA_VISIBLE_DEVICES'
        for idx, key in enumerate(config['container']['environment']):
            if target_key in key:
                config['container']['environment'][idx] = f"{target_key}={gpu}"
                # print(config['container']['environment'][idx]), exit(0)
                break
        # save config in tmp file
        tmp_file = f'tmp-gpu-{gpu}.yaml'
        with open(tmp_file, 'w') as f:
            yaml.dump(config, f)

        # run experiment detached
        print(f'Running experiment on GPU {gpu}')

        # detach output
        subprocess.Popen(['crafting', f'{tmp_file}'],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)
        time.sleep(2)

    for gpu in gpu_number.split(','):
        tmp_file = f'tmp-gpu-{gpu}.yaml'
        os.remove(tmp_file)


def main():
    os.system("git stash && git pull")
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpus', type=str, default='0,1')
    run_experiment(gpu_number=parser.parse_args().use_gpus)


if __name__ == '__main__':
    main()
