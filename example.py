#!/usr/bin/env python3
"""
Example usage of test_fastpd library
"""

import numpy as np


def create_simple_example():
    """Create a simple example for testing FastPD"""

    # Simple 2x2 grid, each node has 2 labels
    num_nodes = 4
    num_labels = 2

    # Create unary terms - the cost of each label for each node
    # Shape: (num_nodes * num_labels,)
    unary = np.array(
        [
            1.0,
            2.0,  # Node 0: label 0 cost 1.0, label 1 cost 2.0
            2.0,
            1.0,  # Node 1: label 0 cost 2.0, label 1 cost 1.0
            1.5,
            1.5,  # Node 2: label 0 cost 1.5, label 1 cost 1.5
            2.0,
            1.0,  # Node 3: label 0 cost 2.0, label 1 cost 1.0
        ],
        dtype=np.float32,
    )

    # Create pairs - pairs of connected nodes
    # Connections for a 2x2 grid: (0,1), (0,2), (1,3), (2,3)
    pairs = np.array(
        [
            0,
            1,  # Edge 0: connects node 0 and 1
            0,
            2,  # Edge 1: connects node 0 and 2
            1,
            3,  # Edge 2: connects node 1 and 3
            2,
            3,  # Edge 3: connects node 2 and 3
        ],
        dtype=np.int32,
    )

    # Create binary terms - the cost of label pairs for each edge
    # Each edge has a num_labels x num_labels cost matrix
    binary_cost = np.array(
        [[0.0, 1.0], [1.0, 0.0]], dtype=np.float32  # Same label cost 0, different label cost 1
    ).flatten()

    # Create the same binary cost for each edge
    binary_list = [binary_cost.copy() for _ in range(len(pairs) // 2)]

    return unary, binary_list, pairs


def main():
    """Main example function"""
    print("FastPD Example")
    print("=" * 50)

    try:
        import spateo

        print("✓ Successfully imported spateo")
    except ImportError as e:
        print(f"✗ Failed to import spateo: {e}")
        print("Please run 'pip install -e .' first to build the package.")
        return

    # Check if the fastpd function is available
    if not hasattr(spateo, "libfastpd"):
        print("✗ fastpd function not available")
        return

    print("✓ fastpd function is available")

    # Create sample data
    print("\nCreating example data...")
    unary, binary_list, pairs = create_simple_example()

    print(f"Unary shape: {unary.shape}")
    print(f"Number of binary terms: {len(binary_list)}")
    print(f"Pairs shape: {pairs.shape}")
    print(f"Unary data: {unary}")
    print(f"Pairs: {pairs}")

    # Run the FastPD algorithm
    print("\nRunning FastPD algorithm...")
    try:
        num_iterations = 10
        result = spateo.libfastpd.fastpd(unary, binary_list, pairs, num_iterations)
        print(f"✓ FastPD completed successfully!")
        print(f"Result labeling: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")

    except Exception as e:
        print(f"✗ Error running FastPD: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
