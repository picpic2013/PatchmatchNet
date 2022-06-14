import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')

    parser.add_argument('--device', default='cpu', help='device to run program(cpu / cuda:0 / cuda:1 / ...)')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print(args.opts)