#!/usr/bin/env python
import model
import argparse

from dlc_practical_prologue import generate_pair_sets

# Handle flags

def main():
    # Initialize data
    data = generate_pair_sets(args.data_size)
    print("Generated dataset with {} samples")

    # Initialize models
    models = [
        Model([], data),
    ]

    # Train models
    for model in models:
        model.train()

    # Generate performance metric reports / graphs

if __name__ == '__main__':
    main()
