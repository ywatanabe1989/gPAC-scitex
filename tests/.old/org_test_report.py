#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate org-mode test reports from pytest results.

This script runs pytest on specified tests and generates a nicely formatted
org-mode report with test results, execution time, and error details if any.

Usage:
    python org_test_report.py [test_path]
    
    If test_path is not provided, runs all tests.
    Examples:
    - python org_test_report.py
    - python org_test_report.py tests/
    - python org_test_report.py tests/custom/test_integration_full_pipeline.py
    - python org_test_report.py tests/custom/test_integration_full_pipeline.py::TestFullPACPipeline::test_permutation_testing
"""

import os
import sys
import re
import subprocess
from datetime import datetime


def remove_ansi_escape_codes(text):
    """Remove ANSI escape codes from text."""
    ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape_pattern.sub('', text)


def extract_test_time(stdout):
    """Extract test execution time from pytest output."""
    execution_time = 0.0
    for line in stdout.split('\n'):
        if " in " in line and "s" in line and any(x in line for x in ["passed", "failed", "skipped"]):
            try:
                time_part = line.split("in ")[1].split('s')[0].strip()
                execution_time = float(time_part)
                return execution_time
            except:
                pass
    return execution_time


def parse_test_results(stdout):
    """Parse pytest output to extract test results."""
    # Clean stdout by removing ANSI codes
    clean_stdout = remove_ansi_escape_codes(stdout)
    
    # Extract test counts
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Look for summary line like "4 passed, 1 failed, 2 skipped in 3.45s"
    summary_pattern = r'(\d+) passed,? (\d+) failed,? (\d+) skipped'
    summary_match = re.search(summary_pattern, clean_stdout)
    if summary_match:
        passed_count = int(summary_match.group(1))
        failed_count = int(summary_match.group(2))
        skipped_count = int(summary_match.group(3))
    else:
        # Try individual patterns
        passed_pattern = r'(\d+) passed'
        failed_pattern = r'(\d+) failed'
        skipped_pattern = r'(\d+) skipped'
        
        passed_match = re.search(passed_pattern, clean_stdout)
        if passed_match:
            passed_count = int(passed_match.group(1))
        
        failed_match = re.search(failed_pattern, clean_stdout)
        if failed_match:
            failed_count = int(failed_match.group(1))
        
        skipped_match = re.search(skipped_pattern, clean_stdout)
        if skipped_match:
            skipped_count = int(skipped_match.group(1))
    
    # Extract individual test results
    passed_tests = []
    failed_tests = []
    skipped_tests = []
    
    # Process each line for test results
    collected_testfiles = []
    for line in clean_stdout.split("\n"):
        if line.startswith("collecting ") or "collecting" not in line:
            # Skip collecting lines without tests
            if ".py" in line and "::" in line and any(x in line for x in ["PASSED", "FAILED", "SKIPPED"]):
                # This is a test result line
                nodeid = line.strip().split()[0]  # Extract the test node ID
                status = "UNKNOWN"
                if "PASSED" in line:
                    status = "PASSED"
                elif "FAILED" in line:
                    status = "FAILED"
                elif "SKIPPED" in line:
                    status = "SKIPPED"
                
                # Parse the node ID
                parts = nodeid.split("::")
                file_path = parts[0]
                test_name = "::".join(parts[1:]) if len(parts) > 1 else file_path
                
                test_info = {
                    'file': file_path,
                    'test': test_name,
                    'status': status,
                    'details': ''
                }
                
                if status == "PASSED":
                    passed_tests.append(test_info)
                elif status == "FAILED":
                    failed_tests.append(test_info)
                    # Try to extract failure details
                    if test_name in clean_stdout:
                        start_idx = clean_stdout.find(test_name)
                        end_idx = clean_stdout.find("FAILED", start_idx)
                        if end_idx > start_idx:
                            test_info['details'] = clean_stdout[start_idx:end_idx].strip()
                elif status == "SKIPPED":
                    skipped_tests.append(test_info)
            elif ".py" in line and ("collected " not in line or "no tests" not in line):
                # This might be a testfile
                testfile = line.strip()
                if os.path.exists(testfile) and testfile.endswith(".py"):
                    collected_testfiles.append(testfile)
    
    # Extract failure details
    if failed_count > 0:
        # Find the FAILURES section
        failures_section = ""
        if "FAILURES" in clean_stdout:
            failures_parts = clean_stdout.split("FAILURES")
            if len(failures_parts) > 1:
                failures_section = failures_parts[1]
                # Find where the failures section ends
                end_markers = ["warnings summary", "short test summary info", "==="]
                for marker in end_markers:
                    if marker in failures_section:
                        failures_section = failures_section.split(marker)[0]
        
        # Associate failure details with each failed test
        for test in failed_tests:
            test_name = test['test']
            if test_name in failures_section:
                # Extract the specific test's failure details
                test_start = failures_section.find(test_name)
                if test_start >= 0:
                    test_end = len(failures_section)
                    # Find the end of this test's failure details
                    end_test_markers = ["___", "==="]
                    for marker in end_test_markers:
                        next_marker = failures_section.find(marker, test_start + len(test_name))
                        if next_marker > test_start:
                            test_end = next_marker
                            break
                    
                    test['details'] = failures_section[test_start:test_end].strip()
    
    return {
        'passed': passed_tests,
        'failed': failed_tests,
        'skipped': skipped_tests,
        'passed_count': passed_count,
        'failed_count': failed_count,
        'skipped_count': skipped_count,
        'total_count': passed_count + failed_count + skipped_count,
        'execution_time': extract_test_time(clean_stdout),
        'collected_testfiles': collected_testfiles
    }


def generate_org_report(test_info, project_name="gPAC"):
    """Generate an org-mode formatted test report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Group tests by file
    tests_by_file = {}
    for status in ['passed', 'failed', 'skipped']:
        for test in test_info[status]:
            file_path = test['file']
            if file_path not in tests_by_file:
                tests_by_file[file_path] = {
                    'passed': [],
                    'failed': [],
                    'skipped': []
                }
            tests_by_file[file_path][status].append(test)
    
    # Calculate success rate
    total = test_info['total_count']
    passed = test_info['passed_count']
    success_rate = (passed / total * 100) if total > 0 else 0
    
    # Generate the report
    org_lines = []
    org_lines.append(f"#+TITLE: {project_name} Test Report")
    org_lines.append(f"#+AUTHOR: Generated by org_test_report.py")
    org_lines.append(f"#+DATE: {now}")
    org_lines.append("")
    
    # Test results summary
    org_lines.append("* Test Results Summary")
    org_lines.append(f"- Passed: {test_info['passed_count']}")
    org_lines.append(f"- Failed: {test_info['failed_count']}")
    org_lines.append(f"- Skipped: {test_info['skipped_count']}")
    org_lines.append(f"- Total: {test_info['total_count']}")
    org_lines.append(f"- Total Time: {test_info['execution_time']} seconds")
    org_lines.append(f"- Success Rate: {success_rate:.1f}%")
    org_lines.append("")
    
    # Passed tests section
    if test_info['passed']:
        org_lines.append(f"* Passed Tests ({test_info['passed_count']})")
        org_lines.append("")
        
        # Group by file
        for file_path, tests in sorted(tests_by_file.items()):
            if tests['passed']:
                org_lines.append(f"** {file_path} ({len(tests['passed'])} tests)")
                for test in tests['passed']:
                    test_name = test['test']
                    org_lines.append(f"- [[file:{file_path}::{test_name}][{test_name}]]")
                org_lines.append("")
    
    # Failed tests section
    if test_info['failed']:
        org_lines.append(f"* Failed Tests ({test_info['failed_count']})")
        org_lines.append("")
        
        # Group by file
        for file_path, tests in sorted(tests_by_file.items()):
            if tests['failed']:
                org_lines.append(f"** {file_path} ({len(tests['failed'])} tests)")
                for test in tests['failed']:
                    test_name = test['test']
                    org_lines.append(f"- [[file:{file_path}::{test_name}][{test_name}]]")
                    
                    # Add error details if available
                    if test['details']:
                        org_lines.append("  + Error details:")
                        for line in test['details'].split('\n'):
                            org_lines.append(f"    {line}")
                org_lines.append("")
    
    # Skipped tests section
    if test_info['skipped']:
        org_lines.append(f"* Skipped Tests ({test_info['skipped_count']})")
        org_lines.append("")
        
        # Group by file
        for file_path, tests in sorted(tests_by_file.items()):
            if tests['skipped']:
                org_lines.append(f"** {file_path} ({len(tests['skipped'])} tests)")
                for test in tests['skipped']:
                    test_name = test['test']
                    org_lines.append(f"- [[file:{file_path}::{test_name}][{test_name}]]")
                org_lines.append("")
    
    return '\n'.join(org_lines)


def run_tests(test_path=None):
    """Run pytest on specified test_path or all tests."""
    cmd = ["python", "-m", "pytest", "-v"]
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    # Include module import related arguments
    cmd.extend(["--import-mode=importlib"])
    
    # Set environment variable to include project root in Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.getcwd())
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            check=False
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return f"Error running pytest: {e}", "", 1


def main():
    """Main function to run tests and generate report."""
    # Check if a specific test path was provided
    test_path = None
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    
    # Run the tests
    stdout, stderr, exit_code = run_tests(test_path)
    
    if "Error running pytest" in stdout:
        print(f"Error: {stdout}")
        return 1
    
    # Parse test results
    test_info = parse_test_results(stdout)
    
    # Generate org report
    org_report = generate_org_report(test_info)
    
    # Determine the report filename
    if test_path:
        # Generate a report name based on the test path
        base_name = os.path.basename(test_path.split("::")[0])
        report_name = f"test_report_{base_name.replace('.py', '')}.org"
    else:
        report_name = "test_report.org"
    
    report_path = os.path.join(os.getcwd(), report_name)
    
    # Write the report
    with open(report_path, 'w') as f:
        f.write(org_report)
    
    print(f"Test report generated at {report_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())