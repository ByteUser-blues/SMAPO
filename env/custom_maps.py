import re
from pathlib import Path

import yaml

with open(Path(__file__).parent / "maps.yaml", "r") as f:
    maps = yaml.safe_load(f)

with open(Path(__file__).parent / "random-pico.yaml", "r") as f:
    random_pico = yaml.safe_load(f)

with open(Path(__file__).parent / "random.yaml", "r") as f:
    random_map = yaml.safe_load(f)

with open(Path(__file__).parent / "street_map.yaml", "r") as f:
    street = yaml.safe_load(f)

maps = {**maps, **random_pico, **random_map, **street}

MAPS_REGISTRY = maps
_test_regexp = '(wc3-[A-P]|sc1-[A-S]|sc1-TaleofTwoCities|street-[A-P]|mazes-s[0-9]_|mazes-s[1-3][0-9]_|random-s[0-9]_|random-s[1-3][0-9]_)'


def split_train_test():
    with open(Path(__file__).parent / "maps.yaml", "r") as f:
        maps = yaml.safe_load(f)
    collections = dict()
    for m in maps:
        if m.split('-')[0] not in collections:
            collections[m.split('-')[0]] = [m]
        else:
            collections[m.split('-')[0]].append(m)
    train = []
    test = []
    for c in collections:
        i = 0
        while (i < len(collections[c]) * 0.8):
            train.append(collections[c][i])
            i += 1
        while i < len(collections[c]):
            test.append(collections[c][i])
            i += 1

    for name in train:
        assert re.match(_test_regexp, name), f'{name} must be in train'

    for name in test:
        assert not re.match(_test_regexp, name), f'{name} must not be in train'


def main():
    with open(Path(__file__).parent / "maps.yaml", "r") as f:
        maps = yaml.safe_load(f)

    train = []
    test = []
    for name in maps:
        if re.match(_test_regexp, name):
            train.append(name)
        else:
            test.append(name)
    print(train)
    print(test)


if __name__ == '__main__':
    main()
