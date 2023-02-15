from cellular_automata.Lenia import Lenia







def main():
    print("Welcome to my program!")
    # Add your code here


if __name__ == "__main__":
    args = {'board': None, 'board_size': 64, 'kernel_size': 16, 'kernel_peaks': None, 'mu': 0.25, 'sigma': 0.5, 'dt': 0.1, 'frames': 100, 'seed': None}
    lenia = Lenia(args)
    print(lenia.run_simulation())