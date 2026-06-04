
# src/experiments/frozenlake_generate_map_example.py

import gymnasium.envs.toy_text.frozen_lake as fl

def generate_and_print_map(size=8, p=0.8):
    # Generiert eine zufällige Karte
    random_map = fl.generate_random_map(size=size, p=p)
    print("Generated FrozenLake map:")
    for row in random_map:
        print(row)

    print("\nCopy-paste Python list:")
    print("[")
    for row in random_map:
        print(f'    "{row}",')
    print("]")

if __name__ == "__main__":
    generate_and_print_map(size=6, p=0.8)

