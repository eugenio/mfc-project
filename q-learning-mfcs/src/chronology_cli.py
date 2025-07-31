#!/usr/bin/env python3
"""
Simulation Chronology CLI

Command-line interface for managing simulation chronology records.
Provides commands to view, export, import, and manage simulation history.

Usage:
    python chronology_cli.py --help
    python chronology_cli.py summary
    python chronology_cli.py list --recent 10
    python chronology_cli.py export --format yaml --output exports/chronology.yaml
    python chronology_cli.py import --file chronology_backup.yaml

Created: 2025-07-31
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.simulation_chronology import get_chronology_manager


def cmd_summary(args) -> None:
    """Display chronology summary."""
    manager = get_chronology_manager(args.chronology_file)
    summary = manager.get_chronology_summary()
    
    print("📊 SIMULATION CHRONOLOGY SUMMARY")
    print("=" * 50)
    print(f"Total Simulations: {summary['total_entries']}")
    print(f"Successful Runs: {summary['successful_runs']}")
    print(f"Failed Runs: {summary['failed_runs']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Simulation Time: {summary['total_simulation_time_hours']:.2f} hours")
    print(f"Total Execution Time: {summary['total_execution_time_seconds']:.2f} seconds")
    print(f"Created: {summary['created_at']}")
    print(f"Last Updated: {summary['last_updated']}")
    
    if summary['unique_tags']:
        print(f"\nTags Used: {', '.join(summary['unique_tags'])}")


def cmd_list(args) -> None:
    """List simulation entries."""
    manager = get_chronology_manager(args.chronology_file)
    
    if args.recent:
        entries = manager.chronology.get_recent_entries(args.recent)
        print(f"📋 RECENT {len(entries)} SIMULATIONS")
    else:
        entries = manager.chronology.entries
        print(f"📋 ALL {len(entries)} SIMULATIONS")
    
    print("=" * 80)
    
    for entry in entries:
        status = "✅" if entry.success else "❌"
        tags_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
        
        print(f"{status} {entry.id} | {entry.simulation_name}{tags_str}")
        print(f"    {entry.timestamp} | {entry.duration_hours:.1f}h | {entry.execution_time_seconds:.1f}s")
        if entry.description:
            print(f"    {entry.description}")
        
        # Show key results
        if entry.results_summary:
            key_metrics = []
            for key in ['total_energy', 'average_power', 'coulombic_efficiency']:
                if key in entry.results_summary:
                    value = entry.results_summary[key]
                    if isinstance(value, (int, float)):
                        key_metrics.append(f"{key}: {value:.3f}")
            if key_metrics:
                print(f"    {' | '.join(key_metrics)}")
        
        print()


def cmd_show(args) -> None:
    """Show detailed information about a specific entry."""
    manager = get_chronology_manager(args.chronology_file)
    entry = manager.chronology.get_entry_by_id(args.entry_id)
    
    if not entry:
        print(f"❌ Entry not found: {args.entry_id}")
        return
    
    print(f"🔍 SIMULATION ENTRY: {entry.id}")
    print("=" * 50)
    print(f"Name: {entry.simulation_name}")
    print(f"Description: {entry.description}")
    print(f"Timestamp: {entry.timestamp}")
    print(f"Duration: {entry.duration_hours:.2f} hours")
    print(f"Execution Time: {entry.execution_time_seconds:.2f} seconds")
    print(f"Success: {'✅ Yes' if entry.success else '❌ No'}")
    
    if entry.error_message:
        print(f"Error: {entry.error_message}")
    
    if entry.tags:
        print(f"Tags: {', '.join(entry.tags)}")
    
    print("\n📊 RESULTS SUMMARY:")
    if entry.results_summary:
        for key, value in entry.results_summary.items():
            print(f"  {key}: {value}")
    else:
        print("  No results summary available")
    
    if entry.result_files:
        print("\n📁 RESULT FILES:")
        for file_type, path in entry.result_files.items():
            print(f"  {file_type}: {path}")
    
    if entry.qlearning_config:
        print("\n⚙️  Q-LEARNING CONFIG:")
        print("  Configuration available (use --show-config for details)")
    
    if entry.sensor_config:
        print("\n📡 SENSOR CONFIG:")
        print("  Configuration available (use --show-config for details)")


def cmd_export(args) -> None:
    """Export chronology to file."""
    manager = get_chronology_manager(args.chronology_file)
    output_path = Path(args.output)
    
    if args.format.lower() == 'yaml':
        manager.export_chronology_yaml(output_path)
    elif args.format.lower() == 'json':
        manager.export_chronology_json(output_path)
    else:
        print(f"❌ Unsupported format: {args.format}")
        return
    
    print(f"✅ Chronology exported to: {output_path}")


def cmd_import(args) -> None:
    """Import chronology from file."""
    input_path = Path(args.file)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    if not args.force:
        response = input("⚠️  This will replace the current chronology. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Import cancelled")
            return
    
    manager = get_chronology_manager(args.chronology_file)
    
    try:
        manager.import_chronology_yaml(input_path)
        print(f"✅ Chronology imported from: {input_path}")
    except Exception as e:
        print(f"❌ Import failed: {e}")


def cmd_tags(args) -> None:
    """List all tags or show entries with specific tag."""
    manager = get_chronology_manager(args.chronology_file)
    
    if args.tag:
        entries = manager.chronology.get_entries_by_tag(args.tag)
        print(f"📋 ENTRIES WITH TAG '{args.tag}' ({len(entries)} found)")
        print("=" * 50)
        
        for entry in entries:
            status = "✅" if entry.success else "❌"
            print(f"{status} {entry.id} | {entry.simulation_name}")
            print(f"    {entry.timestamp}")
            if entry.description:
                print(f"    {entry.description}")
            print()
    else:
        # Show all tags
        all_tags = set()
        tag_counts = {}
        
        for entry in manager.chronology.entries:
            for tag in entry.tags:
                all_tags.add(tag)
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        print(f"🏷️  ALL TAGS ({len(all_tags)} unique)")
        print("=" * 30)
        
        for tag in sorted(all_tags):
            print(f"{tag} ({tag_counts[tag]} entries)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Simulation Chronology Manager")
    parser.add_argument('--chronology-file', default='simulation_chronology.yaml',
                       help='Path to chronology file (default: simulation_chronology.yaml)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summary command
    subparsers.add_parser('summary', help='Show chronology summary')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List simulation entries')
    list_parser.add_argument('--recent', type=int, metavar='N',
                           help='Show only N most recent entries')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show detailed entry information')
    show_parser.add_argument('entry_id', help='Entry ID to show')
    show_parser.add_argument('--show-config', action='store_true',
                           help='Show full configuration details')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export chronology')
    export_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                             help='Export format (default: yaml)')
    export_parser.add_argument('--output', required=True,
                             help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import chronology')
    import_parser.add_argument('--file', required=True,
                             help='Input file path')
    import_parser.add_argument('--force', action='store_true',
                             help='Skip confirmation prompt')
    
    # Tags command
    tags_parser = subparsers.add_parser('tags', help='Manage tags')
    tags_parser.add_argument('--tag', help='Show entries with specific tag')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    command_map = {
        'summary': cmd_summary,
        'list': cmd_list,
        'show': cmd_show,
        'export': cmd_export,
        'import': cmd_import,
        'tags': cmd_tags
    }
    
    command_func = command_map.get(args.command)
    if command_func:
        try:
            command_func(args)
        except Exception as e:
            print(f"❌ Error executing command: {e}")
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()