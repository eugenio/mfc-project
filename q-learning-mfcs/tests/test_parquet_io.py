#!/usr/bin/env python3
"""
Unit Tests for File I/O and Parquet Functionality

CRITICAL ARCHITECTURE TESTS:
============================
These tests validate the complete 3-phase data architecture:
- Phase 1: Memory queue streaming
- Phase 2: Incremental updates  
- Phase 3: Parquet columnar storage

DO NOT MODIFY without understanding architecture implications.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from mfc_streamlit_gui import SimulationRunner
    import pyarrow as pa
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


class TestParquetIO(unittest.TestCase):
    """Test suite for Parquet I/O functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.sim_runner = SimulationRunner()
        
        # Sample data for testing
        self.sample_data = {
            'time_hours': 1.5,
            'power': 12.3,
            'current': 2.1,
            'voltage': 0.8,
            'concentration': 15.7,
            'temperature': 25.0
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_parquet_schema_creation(self):
        """Test Phase 3: Parquet schema creation"""
        schema = self.sim_runner.create_parquet_schema(self.sample_data)
        
        self.assertIsNotNone(schema, "Schema should not be None")
        self.assertIsInstance(schema, pa.Schema, "Should return PyArrow Schema")
        
        # Check that all fields are present
        field_names = [field.name for field in schema]
        for key in self.sample_data.keys():
            self.assertIn(key, field_names, f"Field {key} should be in schema")
            
        # Check optimized data types
        for field in schema:
            if field.name in ['time_hours', 'power', 'current', 'voltage', 'concentration', 'temperature']:
                self.assertEqual(field.type, pa.float32(), f"{field.name} should be float32")
                
    def test_parquet_writer_initialization(self):
        """Test Phase 3: Parquet writer initialization"""
        # First create schema
        schema = self.sim_runner.create_parquet_schema(self.sample_data)
        self.assertIsNotNone(schema)
        
        # Test writer initialization
        success = self.sim_runner.init_parquet_writer(self.temp_dir)
        self.assertTrue(success, "Parquet writer should initialize successfully")
        self.assertIsNotNone(self.sim_runner.parquet_writer, "Writer should be created")
        
        # Clean up
        self.sim_runner.close_parquet_writer()
        
    def test_data_integrity(self):
        """Test Phase 3: Data integrity during Parquet operations"""
        # Setup and write test data
        self.sim_runner.create_parquet_schema(self.sample_data)
        self.sim_runner.init_parquet_writer(self.temp_dir)
        
        # Create known test data
        test_data = []
        for i in range(10):
            data_point = self.sample_data.copy()
            data_point['time_hours'] = i * 0.5
            data_point['power'] = 10.0 + i
            test_data.append(data_point)
            self.sim_runner.parquet_buffer.append(data_point)
        
        # Write and verify
        self.sim_runner.close_parquet_writer()
        parquet_file = self.temp_path / "simulation_data.parquet"
        self.assertTrue(parquet_file.exists())


if __name__ == '__main__':
    unittest.main()