#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 04:20:01 (ywatanabe)"
# File: /home/ywatanabe/proj/gPAC/scripts/exp_01_synthetic_data_preparation/test_data_generation.py
# ----------------------------------------
import os
__FILE__ = (
    "./scripts/exp_01_synthetic_data_preparation/test_data_generation.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Functionalities:
  - Tests the synthetic PAC data generation process
  - Verifies correct class distribution and signal properties
  - Checks if dataset splits are properly created
  - Validates PyTorch Dataset functionality
  - Ensures visualization files are generated correctly

Dependencies:
  - packages:
    - mngs
    - numpy
    - torch
    - matplotlib

IO:
  - input-files:
    - ./scripts/exp_01_synthetic_data_preparation/data/synthetic_pac_signals.pt
    - ./scripts/exp_01_synthetic_data_preparation/data/train_dataset.pt
    - ./scripts/exp_01_synthetic_data_preparation/data/val_dataset.pt
    - ./scripts/exp_01_synthetic_data_preparation/data/test_dataset.pt
"""

"""Imports"""
import argparse
from pathlib import Path
import numpy as np
import torch
import mngs
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

# Access project root
project_root = Path(__FILE__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the data generation class
sys.path.append(str(Path(__DIR__).resolve()))
from generate_synthetic_data import SyntheticPACDataset

"""Parameters"""
DEFAULT_PARAMS = {
    'data_dir': './scripts/exp_01_synthetic_data_preparation/data',
    'small_test': True,  # Run a smaller test with reduced samples for quick testing
}

"""Functions & Classes"""
def test_dataset_basic_properties(dataset_path):
    """Test basic properties of the main dataset."""
    print(f"Testing dataset at {dataset_path}")
    
    # Load dataset
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset file not found at {dataset_path}")
        return False
        
    try:
        data = torch.load(dataset_path)
        print("✓ Successfully loaded dataset")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return False
    
    # Check general structure
    try:
        signals = data['signals']
        metadata = data['metadata']
        params = data['params']
        print("✓ Dataset has correct structure (signals, metadata, params)")
    except KeyError as e:
        print(f"ERROR: Missing expected key in dataset: {e}")
        return False
    
    # Check signals shape and type
    try:
        assert isinstance(signals, torch.Tensor), "Signals should be a torch.Tensor"
        assert len(signals.shape) == 4, f"Expected 4D tensor, got {len(signals.shape)}D"
        assert signals.dtype == torch.float32, f"Expected float32 tensor, got {signals.dtype}"
        print(f"✓ Signals tensor has correct shape {signals.shape} and dtype {signals.dtype}")
    except AssertionError as e:
        print(f"ERROR: {e}")
        return False
    
    # Check metadata
    try:
        class_labels = metadata['class_labels']
        assert len(class_labels) == signals.shape[0], "Class labels count mismatch"
        assert len(torch.unique(class_labels)) == 5, "Expected 5 distinct classes"
        print(f"✓ Class labels have correct shape {class_labels.shape} with 5 classes")
        
        # Check class distribution
        class_counts = torch.bincount(class_labels)
        print(f"Class distribution: {class_counts}")
        assert torch.all(class_counts > 0), "Some classes have no samples"
        
        # Check other metadata fields
        expected_fields = ['pha_freqs', 'amp_freqs', 'noise_levels', 'coupling_strengths', 'sample_ids']
        for field in expected_fields:
            assert field in metadata, f"Missing metadata field: {field}"
            assert len(metadata[field]) == len(signals), f"Metadata field {field} size mismatch"
        print(f"✓ All expected metadata fields present with correct sizes")
        
    except AssertionError as e:
        print(f"ERROR: {e}")
        return False
    
    # Verify frequency ranges match expected values
    try:
        for class_id in range(5):
            mask = class_labels == class_id
            pha_freqs = metadata['pha_freqs'][mask]
            amp_freqs = metadata['amp_freqs'][mask]
            
            # Check frequency ranges for each class
            if class_id == 0:  # Class 1: Low phase + Low amp
                assert torch.all((pha_freqs >= 4.0) & (pha_freqs <= 8.0)), "Class 1 phase freq range error"
                assert torch.all((amp_freqs >= 50.0) & (amp_freqs <= 70.0)), "Class 1 amp freq range error"
            elif class_id == 1:  # Class 2: Low phase + High amp
                assert torch.all((pha_freqs >= 4.0) & (pha_freqs <= 8.0)), "Class 2 phase freq range error"
                assert torch.all((amp_freqs >= 150.0) & (amp_freqs <= 170.0)), "Class 2 amp freq range error"
            elif class_id == 2:  # Class 3: Medium phase + Medium amp
                assert torch.all((pha_freqs >= 9.0) & (pha_freqs <= 14.0)), "Class 3 phase freq range error"
                assert torch.all((amp_freqs >= 100.0) & (amp_freqs <= 120.0)), "Class 3 amp freq range error"
            elif class_id == 3:  # Class 4: High phase + Low amp
                assert torch.all((pha_freqs >= 15.0) & (pha_freqs <= 20.0)), "Class 4 phase freq range error"
                assert torch.all((amp_freqs >= 50.0) & (amp_freqs <= 70.0)), "Class 4 amp freq range error"
            elif class_id == 4:  # Class 5: High phase + High amp
                assert torch.all((pha_freqs >= 15.0) & (pha_freqs <= 20.0)), "Class 5 phase freq range error"
                assert torch.all((amp_freqs >= 150.0) & (amp_freqs <= 170.0)), "Class 5 amp freq range error"
        
        print("✓ Frequency ranges for all classes match expected values")
    except AssertionError as e:
        print(f"ERROR: {e}")
        return False
        
    return True

def test_dataset_splits(data_dir):
    """Test dataset splits (train, val, test)."""
    data_dir = Path(data_dir)
    
    # Check if split files exist
    train_path = data_dir / 'train_dataset.pt'
    val_path = data_dir / 'val_dataset.pt'
    test_path = data_dir / 'test_dataset.pt'
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        print("ERROR: One or more dataset split files are missing")
        return False
    
    # Load datasets
    try:
        train_dataset = torch.load(train_path)
        val_dataset = torch.load(val_path)
        test_dataset = torch.load(test_path)
        print("✓ Successfully loaded all dataset splits")
    except Exception as e:
        print(f"ERROR: Failed to load dataset splits: {e}")
        return False
    
    # Verify they are SyntheticPACDataset instances
    try:
        assert isinstance(train_dataset, SyntheticPACDataset), "Train dataset has wrong type"
        assert isinstance(val_dataset, SyntheticPACDataset), "Val dataset has wrong type"
        assert isinstance(test_dataset, SyntheticPACDataset), "Test dataset has wrong type"
        print("✓ All splits are instances of SyntheticPACDataset")
    except AssertionError as e:
        print(f"ERROR: {e}")
        return False
    
    # Check class distribution in each split
    try:
        train_label_counts = torch.bincount(train_dataset.labels.long())
        val_label_counts = torch.bincount(val_dataset.labels.long())
        test_label_counts = torch.bincount(test_dataset.labels.long())
        
        print(f"Train class distribution: {train_label_counts}")
        print(f"Val class distribution: {val_label_counts}")
        print(f"Test class distribution: {test_label_counts}")
        
        # Check all classes are represented in each split
        assert all(count > 0 for count in train_label_counts), "Missing class in train set"
        assert all(count > 0 for count in val_label_counts), "Missing class in val set"
        assert all(count > 0 for count in test_label_counts), "Missing class in test set"
        
        # Check relative split sizes (approximate)
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        train_ratio = len(train_dataset) / total
        val_ratio = len(val_dataset) / total
        test_ratio = len(test_dataset) / total
        
        print(f"Split ratios - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
        assert 0.65 <= train_ratio <= 0.75, f"Train ratio {train_ratio:.2f} outside expected range"
        assert 0.10 <= val_ratio <= 0.20, f"Val ratio {val_ratio:.2f} outside expected range"
        assert 0.10 <= test_ratio <= 0.20, f"Test ratio {test_ratio:.2f} outside expected range"
        
        print("✓ Dataset splits have correct class distribution and sizes")
    except AssertionError as e:
        print(f"ERROR: {e}")
        return False
    
    # Test dataset functionality
    try:
        # Create dataloader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Get a batch
        batch_signals, batch_labels = next(iter(train_loader))
        
        assert batch_signals.shape[0] == 4, f"Unexpected batch size: {batch_signals.shape[0]}"
        assert batch_signals.shape[1:] == train_dataset.signals[0].shape, "Signal shape mismatch in batch"
        assert batch_labels.shape == (4,), "Label shape mismatch in batch"
        
        print("✓ DataLoader successfully created batches from dataset")
    except Exception as e:
        print(f"ERROR: DataLoader test failed: {e}")
        return False
    
    return True

def test_visualizations(data_dir):
    """Test if visualization files were created."""
    data_dir = Path(data_dir)
    
    # Check visualization files
    viz_files = [
        'synthetic_pac_examples.png',
        'frequency_space.png'
    ]
    
    for file in viz_files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"ERROR: Visualization file {file} not found")
            return False
    
    print("✓ All visualization files were created successfully")
    return True

def run_tests(args):
    """Run all tests."""
    data_dir = Path(args.data_dir)
    main_dataset_path = data_dir / 'synthetic_pac_signals.pt'
    
    # Set up simple counter for tests
    total_tests = 3
    passed_tests = 0
    
    # Test 1: Basic dataset properties
    print("\n=== Testing basic dataset properties ===")
    if test_dataset_basic_properties(main_dataset_path):
        passed_tests += 1
    
    # Test 2: Dataset splits
    print("\n=== Testing dataset splits ===")
    if test_dataset_splits(data_dir):
        passed_tests += 1
    
    # Test 3: Visualizations
    print("\n=== Testing visualizations ===")
    if test_visualizations(data_dir):
        passed_tests += 1
    
    # Print summary
    print(f"\n=== Test Summary: {passed_tests}/{total_tests} tests passed ===")
    
    return passed_tests == total_tests

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test synthetic PAC data generation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_PARAMS['data_dir'],
        help="Directory containing the generated data"
    )
    parser.add_argument(
        "--small_test",
        action="store_true",
        default=DEFAULT_PARAMS['small_test'],
        help="Run smaller tests for quick validation"
    )
    
    args = parser.parse_args()
    return args

def run_main() -> None:
    """Initialize mngs framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt
    
    import sys
    import matplotlib.pyplot as plt
    import mngs
    
    args = parse_args()
    
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )
    
    exit_status = 0 if run_tests(args) else 1
    
    mngs.gen.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )

if __name__ == "__main__":
    run_main()

# EOF