#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-14 16:51:20 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/gPAC/tests/custom/test_pip_install_editable.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/custom/test_pip_install_editable.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Tests for verifying editable pip installation."""

import sys
import pkg_resources
import importlib
import pytest


def test_package_is_importable():
    """Test that gpac can be imported."""
    import gpac
    assert gpac is not None
    
    # Also verify the _ModulationIndex module can be imported
    from gpac import _ModulationIndex
    assert _ModulationIndex is not None


def test_package_installed_in_development_mode():
    """Test that gpac is installed in development mode."""
    import gpac
    gpac_path = gpac.__file__
    
    # In development mode, the package should be installed from the src directory
    assert '/src/gpac/' in gpac_path
    
    # Path should be in the current project
    project_dir = os.path.abspath(os.path.join(__DIR__, '../..'))
    assert project_dir in gpac_path, f"Expected {gpac_path} to be in {project_dir}"


def test_return_distribution_option_implemented():
    """Test that the return_distribution option is implemented."""
    from gpac._pac import PAC
    
    # Check if the return_dist parameter exists in the PAC class
    pac = PAC(seq_len=1000, fs=1000, return_dist=True)
    assert hasattr(pac, 'return_dist')
    assert pac.return_dist is True
    
    # Check the same parameter in the calculate_pac function
    from gpac._pac import calculate_pac
    import inspect
    sig = inspect.signature(calculate_pac)
    assert 'return_dist' in sig.parameters
    assert sig.parameters['return_dist'].default is False


if __name__ == '__main__':
    pytest.main(["-v", __file__])