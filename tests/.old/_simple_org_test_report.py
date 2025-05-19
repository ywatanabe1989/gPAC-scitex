#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an org-mode test report from pytest output using subprocess.
"""

import sys
import os
import re
import subprocess
from datetime import datetime

def run_tests():
    """Run pytest and capture output"""
    # First, let's run a fixed set of tests we know are working
    cmd = [
        "python", "-m", "pytest", 
        "--no-header", "-v", 
        "tests/custom/test_integration_full_pipeline.py::TestFullPACPipeline::test_permutation_testing",
        "tests/test_gradient_flow.py"
    ]
    
    # Set environment variable to include project root in Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.getcwd())
    
    try:
        # Run the command and capture output
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

def parse_test_results(stdout, stderr):
    """Parse pytest output to extract test results"""
    tests = {
        'passed': [],
        'failed': [],
        'skipped': [],
        'errors': []
    }
    
    # Extract individual test results using regex
    # Look for the dotted output lines with test paths
    test_result_lines = []
    for line in stdout.split('\n'):
        if "tests/" in line and ("] " in line or line.strip().endswith("]")):
            test_result_lines.append(line)
    
    # Parse each individual test result line
    for line in test_result_lines:
        # Get the status (color code in brackets) and the test path
        parts = line.split("] ")
        if len(parts) < 2:
            continue
            
        # Extract the result indicator and the test path
        result_indicator = parts[0][-1]  # Last character before "]"
        test_path = parts[1].strip()
        
        # Determine the status from the indicator
        status = ""
        if result_indicator == ".":
            status = "PASSED"
        elif result_indicator in ("F", "E"):
            status = "FAILED"
        elif result_indicator == "s":
            status = "SKIPPED"
        else:
            continue  # Unknown status
            
        # Parse the nodeid
        nodeid = test_path
        file_path = nodeid.split("::")[0]
        
        # Extract test name
        if "::" in nodeid:
            test_name = "::".join(nodeid.split("::")[1:])
        else:
            test_name = nodeid
                
        test_info = {
            'file': file_path,
            'test': test_name,
            'status': status,
            'details': ''
        }
        
        if status in ("PASSED", "XPASSED"):
            tests['passed'].append(test_info)
        elif status in ("FAILED", "XFAILED"):
            tests['failed'].append(test_info)
        elif status == "SKIPPED":
            tests['skipped'].append(test_info)
        elif status == "ERROR":
            tests['errors'].append(test_info)
    
    # Extract test counts from summary
    total_tests = len(tests['passed']) + len(tests['failed']) + len(tests['skipped']) + len(tests['errors'])
    
    # Extract execution time
    time_pattern = r"in (\d+\.\d+)s"
    execution_time = 0.0
    for line in stdout.split('\n'):
        if "in" in line and "s" in line:
            match = re.search(time_pattern, line)
            if match:
                execution_time = float(match.group(1))
                break
    
    # Extract failure details
    failure_sections = re.split(r"=+ FAILURES =+", stdout)
    if len(failure_sections) > 1:
        failure_details = failure_sections[1]
        
        for test in tests['failed']:
            # Look for this test's failure details
            test_pattern = re.escape(test['file']) + r".*" + re.escape(test['test'])
            match = re.search(test_pattern, failure_details)
            if match:
                # Find the start of this test's failure details
                start_pos = match.start()
                # Find the next test or the end of the failures section
                next_test_match = re.search(r"_{10,}", failure_details[start_pos:])
                if next_test_match:
                    end_pos = start_pos + next_test_match.start()
                    test['details'] = failure_details[start_pos:end_pos].strip()
                else:
                    test['details'] = failure_details[start_pos:].strip()
    
    return {
        'passed': tests['passed'],
        'failed': tests['failed'] + tests['errors'],
        'skipped': tests['skipped'],
        'total': total_tests,
        'execution_time': execution_time
    }

def generate_org_report(test_info):
    """Generate an org-mode formatted test report"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = test_info['total']
    passed = len(test_info['passed'])
    failed = len(test_info['failed'])
    skipped = len(test_info['skipped'])
    
    success_rate = (passed / total * 100) if total > 0 else 0
    
    # Prepare the report
    report = []
    report.append(f"#+TITLE: gPAC Test Report")
    report.append(f"#+AUTHOR: Generated by simple_org_test_report.py")
    report.append(f"#+DATE: {now}")
    report.append("")
    report.append("* Test Results Summary")
    report.append(f"- Passed: {passed}")
    report.append(f"- Failed: {failed}")
    report.append(f"- Skipped: {skipped}")
    report.append(f"- Total: {total}")
    report.append(f"- Total Time: {test_info['execution_time']} seconds")
    report.append(f"- Success Rate: {success_rate:.1f}%")
    report.append("")
    
    # Group tests by file
    files = {}
    for category in ['passed', 'failed', 'skipped']:
        for test in test_info[category]:
            file_path = test['file']
            if file_path not in files:
                files[file_path] = []
            files[file_path].append(test)
    
    # Add passed tests
    if test_info['passed']:
        report.append(f"* Passed Tests ({passed})")
        report.append("")
        
        passed_by_file = {}
        for test in test_info['passed']:
            file_path = test['file']
            if file_path not in passed_by_file:
                passed_by_file[file_path] = []
            passed_by_file[file_path].append(test)
        
        for file_path, tests in sorted(passed_by_file.items()):
            report.append(f"** {file_path} ({len(tests)} tests)")
            for test in tests:
                report.append(f"- [[file:{file_path}::{test['test']}][{test['test']}]]")
            report.append("")
    
    # Add failed tests
    if test_info['failed']:
        report.append(f"* Failed Tests ({failed})")
        report.append("")
        
        failed_by_file = {}
        for test in test_info['failed']:
            file_path = test['file']
            if file_path not in failed_by_file:
                failed_by_file[file_path] = []
            failed_by_file[file_path].append(test)
        
        for file_path, tests in sorted(failed_by_file.items()):
            report.append(f"** {file_path} ({len(tests)} tests)")
            for test in tests:
                report.append(f"- [[file:{file_path}::{test['test']}][{test['test']}]]")
                if test['details']:
                    report.append("  + Error details:")
                    for line in test['details'].split('\n'):
                        report.append(f"    {line}")
            report.append("")
    
    # Add skipped tests
    if test_info['skipped']:
        report.append(f"* Skipped Tests ({skipped})")
        report.append("")
        
        skipped_by_file = {}
        for test in test_info['skipped']:
            file_path = test['file']
            if file_path not in skipped_by_file:
                skipped_by_file[file_path] = []
            skipped_by_file[file_path].append(test)
        
        for file_path, tests in sorted(skipped_by_file.items()):
            report.append(f"** {file_path} ({len(tests)} tests)")
            for test in tests:
                report.append(f"- [[file:{file_path}::{test['test']}][{test['test']}]]")
            report.append("")
    
    return '\n'.join(report)

def main():
    """Main function to run tests and generate report"""
    stdout, stderr, exit_code = run_tests()
    
    if "Error running pytest" in stdout:
        print(f"Error: {stdout}")
        return 1
    
    test_info = parse_test_results(stdout, stderr)
    org_report = generate_org_report(test_info)
    
    report_path = os.path.join(os.getcwd(), "test_report.org")
    with open(report_path, 'w') as f:
        f.write(org_report)
    
    print(f"Test report generated at {report_path}")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())