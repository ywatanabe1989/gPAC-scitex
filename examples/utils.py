#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for gPAC examples.

This module provides utilities for standardizing example scripts and
handling figure output based on the configuration from run_examples.py.
"""
import os
import inspect
import functools
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Callable, Union

# Get environment variables set by run_examples.py
OUTPUT_DIR = os.environ.get('GPAC_EXAMPLE_OUTPUT_DIR', 'figures')
FORMAT = os.environ.get('GPAC_EXAMPLE_FORMAT', 'png')
DPI = int(os.environ.get('GPAC_EXAMPLE_DPI', 300))

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def configure_matplotlib(interactive: bool = False) -> None:
    """
    Configure matplotlib for example scripts.
    
    Args:
        interactive: Whether to use an interactive backend (default: False)
    """
    import matplotlib
    if not interactive:
        matplotlib.use('Agg')  # Use non-interactive backend
    
    # Set common style settings
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure figure saving parameters
    plt.rcParams['savefig.format'] = FORMAT
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12


def save_figure(fig, name: str, output_dir: Optional[str] = None, 
                format: Optional[str] = None, dpi: Optional[int] = None) -> str:
    """
    Save a figure with standardized settings.
    
    Args:
        fig: Matplotlib figure to save
        name: Name for the saved figure (without extension)
        output_dir: Directory to save to (defaults to environment setting)
        format: File format (defaults to environment setting)
        dpi: Resolution for raster formats (defaults to environment setting)
        
    Returns:
        Full path to the saved figure
    """
    # Use defaults from environment if not specified
    output_dir = output_dir or OUTPUT_DIR
    format = format or FORMAT
    dpi = dpi or DPI
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add extension if not present
    if not name.endswith(f'.{format}'):
        name = f"{name}.{format}"
    
    # Create the full file path
    file_path = os.path.join(output_dir, name)
    
    # Save the figure
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    return file_path


def example_metadata(title: str, description: str, requirements: Optional[List[str]] = None) -> Callable:
    """
    Decorator to add metadata to example functions.
    
    Args:
        title: Short title for the example
        description: Longer description of what the example demonstrates
        requirements: List of required packages beyond core dependencies
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Add metadata to the function
        wrapper.example_title = title
        wrapper.example_description = description
        wrapper.example_requirements = requirements or []
        
        return wrapper
    
    return decorator


class Example:
    """
    Base class for standardized examples.
    
    This class provides a standard interface for examples with 
    convenience methods for common tasks.
    """
    
    def __init__(self, title: str, description: str, 
                 output_prefix: Optional[str] = None):
        """
        Initialize a new example.
        
        Args:
            title: Short title for the example
            description: Longer description of what the example demonstrates
            output_prefix: Prefix for output filenames (defaults to function name)
        """
        self.title = title
        self.description = description
        self._output_prefix = output_prefix or self.__class__.__name__.lower()
        
        # Configure matplotlib
        configure_matplotlib()
    
    def run(self) -> None:
        """Run the example. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the run method")
    
    def save_figure(self, fig, name: str, **kwargs) -> str:
        """
        Save a figure with standard naming.
        
        Args:
            fig: Matplotlib figure to save
            name: Specific name for this figure
            **kwargs: Additional arguments for save_figure
            
        Returns:
            Full path to the saved figure
        """
        # Add output prefix to name
        full_name = f"{self._output_prefix}_{name}"
        return save_figure(fig, full_name, **kwargs)
    
    def print_header(self) -> None:
        """Print a header with the example title and description."""
        print(f"\n{'=' * 80}")
        print(f"{self.title}")
        print(f"{'=' * 80}")
        print(f"{self.description}\n")


def print_example_header(title: str, description: str) -> None:
    """
    Print a formatted header for an example script.
    
    Args:
        title: Example title
        description: Example description
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{description}\n")


# Auto-configure matplotlib at import time
configure_matplotlib()