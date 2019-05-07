#!/usr/bin/env python
from model import *

from arguments import args

def main():
    # Initialize models
    stupidNet = StupidNet1(100)

    # Train models
    stupidNet.train_multiple()

    # Generate performance metric reports / graphs
    print("End of main()")

if __name__ == '__main__':
    main()
