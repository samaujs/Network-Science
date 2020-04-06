import argparse
parser = argparse.ArgumentParser(description='GNN project arguments.')
parser.add_argument('--gpu', dest='gpu', action='store_const',
                    const=True, default=False, help='whether to use GPU.')
args = parser.parse_args()

def main():
    print(args)

    if args.gpu:
        print("gpu : ", args.gpu)
    else:
        print("Using CPU")

if __name__ == "__main__":
    main()

