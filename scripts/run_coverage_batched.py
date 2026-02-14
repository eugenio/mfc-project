"""
Batched pytest coverage runner to avoid OOM issues.

Instead of loading all ~492 test files into one pytest process,
this script runs pytest separately for each test subdirectory,
accumulating coverage data with --cov-append.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    # Change to the tests directory
    tests_dir = Path(__file__).parent.parent / "q-learning-mfcs" / "tests"
    os.chdir(tests_dir)

    # Remove any existing coverage data
    coverage_file = Path(".coverage")
    if coverage_file.exists():
        coverage_file.unlink()
        print(f"Removed existing {coverage_file}")

    # Find all subdirectories (skip __pycache__)
    subdirs = [
        d for d in Path(".").iterdir()
        if d.is_dir() and d.name != "__pycache__" and not d.name.startswith(".")
    ]

    # Find top-level test files
    top_level_tests = list(Path(".").glob("test_*.py"))

    # Track results
    batch_results = []
    failed_batches = []

    print(f"\nFound {len(subdirs)} test subdirectories and {len(top_level_tests)} top-level test files")
    print("=" * 80)

    # Run each subdirectory as a batch
    for subdir in sorted(subdirs):
        # Check if directory has any test files
        test_files = list(subdir.glob("test_*.py"))
        if not test_files:
            print(f"Skipping {subdir.name}/ (no test files)")
            continue

        print(f"\nRunning batch: {subdir.name}/ ({len(test_files)} test files)")
        print("-" * 80)

        cmd = [
            sys.executable,
            "-m", "pytest",
            f"{subdir}/",
            "--cov=../src",
            "--cov-append",
            "-q",
            "--tb=short",
            "--no-header",
            "-p", "no:playwright",
        ]

        result = subprocess.run(cmd, capture_output=False)
        batch_results.append((subdir.name, result.returncode))

        if result.returncode != 0:
            failed_batches.append(subdir.name)
            print(f"✗ FAILED: {subdir.name}/")
        else:
            print(f"✓ PASSED: {subdir.name}/")

    # Run top-level test files if any
    if top_level_tests:
        print(f"\nRunning batch: top-level test files ({len(top_level_tests)} files)")
        print("-" * 80)

        cmd = [
            sys.executable,
            "-m", "pytest",
            *[str(f) for f in top_level_tests],
            "--cov=../src",
            "--cov-append",
            "-q",
            "--tb=short",
            "--no-header",
            "-p", "no:playwright",
        ]

        result = subprocess.run(cmd, capture_output=False)
        batch_results.append(("top-level", result.returncode))

        if result.returncode != 0:
            failed_batches.append("top-level")
            print("✗ FAILED: top-level test files")
        else:
            print("✓ PASSED: top-level test files")

    # Generate coverage reports
    print("\n" + "=" * 80)
    print("Generating coverage reports...")
    print("=" * 80)

    # Text report
    subprocess.run([sys.executable, "-m", "coverage", "report"])

    # HTML report
    subprocess.run([sys.executable, "-m", "coverage", "html"])
    print(f"\nHTML coverage report written to: {tests_dir / 'htmlcov' / 'index.html'}")

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)

    for batch_name, returncode in batch_results:
        status = "✓ PASSED" if returncode == 0 else "✗ FAILED"
        print(f"{status}: {batch_name}")

    # Exit with error if any batch failed
    if failed_batches:
        print(f"\n{len(failed_batches)} batch(es) failed: {', '.join(failed_batches)}")
        return 1
    else:
        print(f"\nAll {len(batch_results)} batches passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())