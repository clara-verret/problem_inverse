import argparse

from monte_carlo_direct import main_monte_carlo


def main() :
    """Compute either the direct or inverse problem"""
    parser = argparse.ArgumentParser(description="parser used to chose which problem to solve")
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group()

    #direct problem
    group.add_argument('-d', '--direct', action='store_true', help="solve direct problem")
    #inverse problem
    group.add_argument('-i', '--inverse', action='store_true', help="solve inverse problem")

    args = parser.parse_args()
    if args.direct:
        print("Solving direct problem")
        main_monte_carlo() # dans le futur mettre la fonction qui construit CLD
    elif args.inverse:
        print("Solving inverse problem")
        #mettre la fonction qui inverse le problème


if __name__ == '__main__':
    main()