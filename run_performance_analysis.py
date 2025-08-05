#!/usr/bin/env python3
"""Simple performance analysis runner."""

import json
import sys
from pathlib import Path

# Add test path
sys.path.insert(0, str(Path(__file__).parent / "q-learning-mfcs" / "tests"))

def main():
    try:
        from gpu.test_performance_coverage import PerformanceCoverageAnalyzer

        print("🚀 Starting Performance Analysis...")
        analyzer = PerformanceCoverageAnalyzer()
        final_report = analyzer.generate_final_report()

        # Save report
        report_file = Path("performance_optimization_report.json")
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        print("\n✅ PERFORMANCE ANALYSIS COMPLETED")
        print("=" * 60)

        # Executive Summary
        exec_summary = final_report["executive_summary"]
        print(f"Overall Status: {exec_summary['overall_status']}")
        print(f"Test Categories: {exec_summary['total_test_categories']}")
        print(f"Coverage Score: {exec_summary['coverage_score']:.1%}")
        print(f"Critical Issues: {exec_summary['critical_issues']}")
        print(f"Recommendations: {exec_summary['recommendations_count']}")

        print(f"\n📋 Report saved to: {report_file.absolute()}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
