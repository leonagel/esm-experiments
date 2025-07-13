#!/usr/bin/env python3
"""
Master test runner for all ESM experiment tests.
Runs all test suites and provides a summary.
"""

import sys
import time
from pathlib import Path

# Import all test modules
sys.path.append(str(Path(__file__).parent))

from test_esm import run_all_tests as run_esm_tests
from test_integration import run_all_integration_tests
from test_models import run_all_model_tests
from test_data import run_all_data_tests
from test_training import run_all_training_tests
from test_analysis import run_all_analysis_tests


def run_test_suite(name, test_func):
    """Run a test suite and return results."""
    print(f"\n{'='*70}")
    print(f"Running {name}")
    print('='*70)
    
    start_time = time.time()
    try:
        success = test_func()
        elapsed = time.time() - start_time
        return {
            'name': name,
            'success': success,
            'elapsed': elapsed,
            'error': None
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'name': name,
            'success': False,
            'elapsed': elapsed,
            'error': str(e)
        }


def main():
    """Run all test suites."""
    print("ESM EXPERIMENT TEST RUNNER")
    print("="*70)
    print("This will run all test suites for the ESM experiment.")
    print("Tests are designed to run quickly on CPU without requiring GPUs.")
    print("="*70)
    
    # Define test suites
    test_suites = [
        ("Core ESM Tests", run_esm_tests),
        ("Integration Tests", run_all_integration_tests),
        ("Model Tests", run_all_model_tests),
        ("Data Tests", run_all_data_tests),
        ("Training Tests", run_all_training_tests),
        ("Analysis Tests", run_all_analysis_tests)
    ]
    
    # Run all tests
    results = []
    total_start = time.time()
    
    for suite_name, suite_func in test_suites:
        result = run_test_suite(suite_name, suite_func)
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"\nTotal test suites: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed:.1f}s")
    
    print("\nDetailed Results:")
    print("-"*70)
    print(f"{'Suite Name':<30} {'Status':<10} {'Time (s)':<10}")
    print("-"*70)
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{result['name']:<30} {status:<10} {result['elapsed']:<10.1f}")
        if result['error']:
            print(f"  Error: {result['error']}")
    
    print("-"*70)
    
    # Exit with appropriate code
    if failed > 0:
        print(f"\n❌ {failed} test suite(s) failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()