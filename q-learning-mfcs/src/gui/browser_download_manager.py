"""
Browser Download Manager for MFC Simulation Data

Implements browser-based download functionality for simulation results,
eliminating the need for server-side file storage and manual navigation.

Features:
- Direct browser downloads via Streamlit download_button
- Multiple format support (CSV, JSON, Excel, HDF5, Parquet)
- Compressed archive downloads for large datasets
- Real-time data streaming during simulation
- Selective data export with preview

Created: 2025-08-02
"""
import io
import json
import zipfile
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


class BrowserDownloadManager:
    """Manages browser-based downloads for simulation data."""
    
    def __init__(self):
        """Initialize download manager."""
        self.format_handlers = {
            'csv': self._to_csv,
            'json': self._to_json,
            'xlsx': self._to_excel,
            'parquet': self._to_parquet,
            'hdf5': self._to_hdf5,
            'zip': self._to_zip_archive
        }
        
    def render_download_interface(
        self,
        data_dict: dict[str, Any],
        simulation_name: str = "mfc_simulation"
    ):
        """
        Render the browser download interface.
        
        Args:
            data_dict: Dictionary of data to download
            simulation_name: Name for the simulation data
        """
        st.markdown("### 📥 Browser Downloads")
        
        if not data_dict:
            st.info("No data available for download. Run a simulation first.")
            return
            
        # Data preview section
        with st.expander("📊 Preview Available Data", expanded=False):
            self._render_data_preview(data_dict)
            
        # Format selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_format = st.selectbox(
                "Select Download Format",
                options=['csv', 'json', 'xlsx', 'parquet', 'hdf5', 'zip'],
                format_func=lambda x: {
                    'csv': '📄 CSV - Universal spreadsheet format',
                    'json': '📋 JSON - Structured data format',
                    'xlsx': '📊 Excel - Microsoft Excel workbook',
                    'parquet': '🚀 Parquet - High-performance columnar',
                    'hdf5': '🗄️ HDF5 - Scientific data format',
                    'zip': '📦 ZIP - Compressed archive (all formats)'
                }.get(x, x),
                key="download_format_selection"
            )
            
        with col2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = st.text_input(
                "Filename",
                value=f"{simulation_name}_{timestamp}",
                key="download_filename"
            )
            
        # Data selection
        st.markdown("#### Select Data to Download")
        
        selected_data = {}
        cols = st.columns(3)
        
        for idx, (name, data) in enumerate(data_dict.items()):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.checkbox(f"📊 {name}", value=True, key=f"download_select_{name}"):
                    selected_data[name] = data
                    
        if not selected_data:
            st.warning("Please select at least one dataset to download.")
            return
            
        # Download options
        with st.expander("⚙️ Advanced Options", expanded=False):
            include_metadata = st.checkbox("Include metadata", value=True)
            compress_large_files = st.checkbox(
                "Compress files larger than 10MB",
                value=True
            )
            
        # Generate download button
        self._render_download_button(
            selected_data,
            selected_format,
            filename,
            include_metadata,
            compress_large_files
        )
        
        # Quick download buttons for common formats
        st.markdown("#### 🚀 Quick Downloads")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download All as CSV", key="quick_csv"):
                self._quick_download(selected_data, 'csv', simulation_name)
                
        with col2:
            if st.button("📋 Download All as JSON", key="quick_json"):
                self._quick_download(selected_data, 'json', simulation_name)
                
        with col3:
            if st.button("📦 Download as ZIP", key="quick_zip"):
                self._quick_download(selected_data, 'zip', simulation_name)
                
    def _render_data_preview(self, data_dict: dict[str, Any]):
        """Render data preview for user inspection."""
        for name, data in data_dict.items():
            st.markdown(f"**{name}**")
            
            if isinstance(data, pd.DataFrame):
                st.write(f"Shape: {data.shape}")
                st.dataframe(data.head())
            elif isinstance(data, dict):
                st.json({k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                        for k, v in list(data.items())[:5]})
            elif isinstance(data, (list, np.ndarray)):
                st.write(f"Length: {len(data)}")
                st.write(data[:5])
            else:
                st.write(f"Type: {type(data).__name__}")
                st.write(str(data)[:200] + "..." if len(str(data)) > 200 else data)
                
    def _render_download_button(
        self,
        data: dict[str, Any],
        format: str,
        filename: str,
        include_metadata: bool,
        compress: bool
    ):
        """Render the main download button."""
        # Prepare data for download
        file_data, mime_type, extension = self._prepare_download(
            data, format, include_metadata
        )
        
        if file_data is None:
            st.error(f"Failed to prepare data for {format} format.")
            return
            
        # Check file size and compress if needed
        file_size = len(file_data)
        size_mb = file_size / (1024 * 1024)
        
        if compress and size_mb > 10 and format != 'zip':
            # Compress the data
            compressed_data = self._compress_data(file_data, f"{filename}.{extension}")
            file_data = compressed_data
            extension = 'zip'
            mime_type = 'application/zip'
            
        # Display file info
        st.info(f"📊 File size: {size_mb:.2f} MB")
        
        # Download button
        st.download_button(
            label=f"⬇️ Download {format.upper()} ({size_mb:.2f} MB)",
            data=file_data,
            file_name=f"{filename}.{extension}",
            mime=mime_type,
            key=f"download_button_{format}",
            help=f"Click to download {filename}.{extension}"
        )
        
    def _prepare_download(
        self,
        data: dict[str, Any],
        format: str,
        include_metadata: bool
    ) -> tuple[bytes | None, str, str]:
        """
        Prepare data for download in specified format.
        
        Returns:
            Tuple of (file_data, mime_type, extension)
        """
        if format not in self.format_handlers:
            return None, '', ''
            
        handler = self.format_handlers[format]
        return handler(data, include_metadata)
        
    def _to_csv(self, data: dict[str, Any], include_metadata: bool) -> tuple[bytes, str, str]:
        """Convert data to CSV format."""
        output = io.BytesIO()
        
        # If multiple datasets, create a zip file
        if len(data) > 1:
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, dataset in data.items():
                    if isinstance(dataset, pd.DataFrame):
                        csv_data = dataset.to_csv(index=False)
                        zf.writestr(f"{name}.csv", csv_data)
                    else:
                        # Convert to DataFrame if possible
                        try:
                            df = pd.DataFrame(dataset)
                            csv_data = df.to_csv(index=False)
                            zf.writestr(f"{name}.csv", csv_data)
                        except:
                            # Save as text
                            zf.writestr(f"{name}.txt", str(dataset))
                            
                if include_metadata:
                    metadata = self._generate_metadata(data)
                    zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                    
            return output.getvalue(), 'application/zip', 'zip'
        else:
            # Single dataset
            name, dataset = list(data.items())[0]
            if isinstance(dataset, pd.DataFrame):
                csv_data = dataset.to_csv(index=False)
            else:
                df = pd.DataFrame(dataset)
                csv_data = df.to_csv(index=False)
                
            return csv_data.encode('utf-8'), 'text/csv', 'csv'
            
    def _to_json(self, data: dict[str, Any], include_metadata: bool) -> tuple[bytes, str, str]:
        """Convert data to JSON format."""
        json_data = {}
        
        for name, dataset in data.items():
            if isinstance(dataset, pd.DataFrame):
                json_data[name] = dataset.to_dict(orient='records')
            elif isinstance(dataset, np.ndarray):
                json_data[name] = dataset.tolist()
            else:
                json_data[name] = dataset
                
        if include_metadata:
            json_data['_metadata'] = self._generate_metadata(data)
            
        json_str = json.dumps(json_data, indent=2, default=str)
        return json_str.encode('utf-8'), 'application/json', 'json'
        
    def _to_excel(self, data: dict[str, Any], include_metadata: bool) -> tuple[bytes, str, str]:
        """Convert data to Excel format."""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for name, dataset in data.items():
                if isinstance(dataset, pd.DataFrame):
                    dataset.to_excel(writer, sheet_name=name[:31], index=False)
                else:
                    try:
                        df = pd.DataFrame(dataset)
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                    except:
                        # Save as text in a sheet
                        df = pd.DataFrame({'data': [str(dataset)]})
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                        
            if include_metadata:
                metadata_df = pd.DataFrame([self._generate_metadata(data)])
                metadata_df.to_excel(writer, sheet_name='metadata', index=False)
                
        return output.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'
        
    def _to_parquet(self, data: dict[str, Any], include_metadata: bool) -> tuple[bytes | None, str, str]:
        """Convert data to Parquet format."""
        if not PARQUET_AVAILABLE:
            st.error("Parquet support not available. Install pyarrow.")
            return None, '', ''
            
        output = io.BytesIO()
        
        # Create a single parquet file with multiple tables
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, dataset in data.items():
                if isinstance(dataset, pd.DataFrame):
                    parquet_buffer = io.BytesIO()
                    dataset.to_parquet(parquet_buffer, index=False)
                    zf.writestr(f"{name}.parquet", parquet_buffer.getvalue())
                    
            if include_metadata:
                metadata = self._generate_metadata(data)
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                
        return output.getvalue(), 'application/zip', 'zip'
        
    def _to_hdf5(self, data: dict[str, Any], include_metadata: bool) -> tuple[bytes | None, str, str]:
        """Convert data to HDF5 format."""
        if not H5PY_AVAILABLE:
            st.error("HDF5 support not available. Install h5py.")
            return None, '', ''
            
        output = io.BytesIO()
        
        # HDF5 requires a file-like object that supports seek
        # For now, save individual HDF5 files in a zip
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, dataset in data.items():
                if isinstance(dataset, pd.DataFrame):
                    hdf_buffer = io.BytesIO()
                    dataset.to_hdf(hdf_buffer, key=name, mode='w')
                    zf.writestr(f"{name}.h5", hdf_buffer.getvalue())
                    
            if include_metadata:
                metadata = self._generate_metadata(data)
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                
        return output.getvalue(), 'application/zip', 'zip'
        
    def _to_zip_archive(self, data: dict[str, Any], include_metadata: bool) -> tuple[bytes, str, str]:
        """Create a ZIP archive with multiple formats."""
        output = io.BytesIO()
        
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save in multiple formats
            for name, dataset in data.items():
                if isinstance(dataset, pd.DataFrame):
                    # CSV
                    csv_data = dataset.to_csv(index=False)
                    zf.writestr(f"csv/{name}.csv", csv_data)
                    
                    # JSON
                    json_data = dataset.to_json(orient='records', indent=2)
                    zf.writestr(f"json/{name}.json", json_data)
                    
            # Add metadata
            if include_metadata:
                metadata = self._generate_metadata(data)
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
                
            # Add README
            readme = self._generate_readme(data)
            zf.writestr("README.txt", readme)
            
        return output.getvalue(), 'application/zip', 'zip'
        
    def _compress_data(self, data: bytes, filename: str) -> bytes:
        """Compress data into a ZIP file."""
        output = io.BytesIO()
        
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, data)
            
        return output.getvalue()
        
    def _generate_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Generate metadata for the dataset."""
        metadata = {
            'created_at': datetime.now().isoformat(),
            'datasets': {},
            'platform': 'MFC Q-Learning Research Platform',
            'version': '2.0'
        }
        
        for name, dataset in data.items():
            if isinstance(dataset, pd.DataFrame):
                metadata['datasets'][name] = {
                    'type': 'DataFrame',
                    'shape': dataset.shape,
                    'columns': list(dataset.columns),
                    'dtypes': {col: str(dtype) for col, dtype in dataset.dtypes.items()}
                }
            else:
                metadata['datasets'][name] = {
                    'type': type(dataset).__name__,
                    'size': len(dataset) if hasattr(dataset, '__len__') else 'N/A'
                }
                
        return metadata
        
    def _generate_readme(self, data: dict[str, Any]) -> str:
        """Generate README file for the archive."""
        readme = f"""MFC Q-Learning Simulation Data Export"""
        return readme
        
    def _quick_download(self, data: dict[str, Any], format: str, name: str):
        """Quick download with default settings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"
        
        file_data, mime_type, extension = self._prepare_download(
            data, format, include_metadata=True
        )
        
        if file_data:
            st.download_button(
                label=f"⬇️ Download {format.upper()}",
                data=file_data,
                file_name=f"{filename}.{extension}",
                mime=mime_type,
                key=f"quick_download_{format}_{timestamp}"
            )


def render_browser_downloads(
    simulation_data: dict[str, Any] | None = None,
    q_learning_data: dict[str, Any] | None = None,
    analysis_results: dict[str, Any] | None = None
):
    """Render browser download interface."""
    download_manager = BrowserDownloadManager()
    
    # Combine all available data
    all_data = {}
    
    if simulation_data:
        all_data.update(simulation_data)
        
    if q_learning_data:
        all_data.update(q_learning_data)
        
    if analysis_results:
        all_data.update(analysis_results)
        
    # Render download interface
    download_manager.render_download_interface(
        all_data,
        simulation_name="mfc_qlearning_results"
    )