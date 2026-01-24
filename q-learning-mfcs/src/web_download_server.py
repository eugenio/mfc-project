"""Web Download Server for Simulation Data.

Provides a lightweight web server to serve simulation result files for browser download
instead of saving to fixed locations. Integrates with the chronology system.

Created: 2025-07-31
"""

from __future__ import annotations

import logging
import mimetypes
import webbrowser
from datetime import datetime
from pathlib import Path

# Try to import streamlit for enhanced web interface
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Try to import FastAPI for API server
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

    # Create dummy classes for type hints when FastAPI not available
    class FastAPI:
        pass


from config.simulation_chronology import get_chronology_manager

logger = logging.getLogger(__name__)


class DownloadServer:
    """Lightweight download server for simulation files."""

    def __init__(
        self,
        output_dir: Path = Path("simulation_outputs"),
        port: int = 8080,
    ) -> None:
        """Initialize download server.

        Args:
            output_dir: Directory containing simulation outputs
            port: Port to run server on

        """
        self.output_dir = Path(output_dir)
        self.port = port
        self.chronology_manager = get_chronology_manager()

        if not FASTAPI_AVAILABLE:
            logger.warning(
                "FastAPI not available. Install with: pip install fastapi uvicorn",
            )

    def create_fastapi_app(self) -> FastAPI | None:
        """Create FastAPI application."""
        if not FASTAPI_AVAILABLE:
            return None

        app = FastAPI(title="MFC Simulation Download Server", version="1.0.0")

        @app.get("/", response_class=HTMLResponse)
        async def download_index():
            """Main download page."""
            return self._generate_download_html()

        @app.get("/api/simulations")
        async def list_simulations():
            """List all simulations with download links."""
            entries = self.chronology_manager.chronology.get_recent_entries(20)

            simulations = []
            for entry in entries:
                sim_data = {
                    "id": entry.id,
                    "name": entry.simulation_name,
                    "timestamp": entry.timestamp,
                    "success": entry.success,
                    "duration_hours": entry.duration_hours,
                    "tags": entry.tags,
                    "download_files": entry.result_files,
                }
                simulations.append(sim_data)

            return {"simulations": simulations}

        @app.get("/api/simulation/{entry_id}")
        async def get_simulation(entry_id: str):
            """Get detailed simulation information."""
            entry = self.chronology_manager.chronology.get_entry_by_id(entry_id)
            if not entry:
                raise HTTPException(status_code=404, detail="Simulation not found")

            return {
                "id": entry.id,
                "name": entry.simulation_name,
                "description": entry.description,
                "timestamp": entry.timestamp,
                "success": entry.success,
                "duration_hours": entry.duration_hours,
                "execution_time_seconds": entry.execution_time_seconds,
                "tags": entry.tags,
                "parameters": entry.parameters,
                "results_summary": entry.results_summary,
                "download_files": entry.result_files,
            }

        @app.get("/download/{entry_id}/{file_type}")
        async def download_file(entry_id: str, file_type: str):
            """Download a specific file for a simulation."""
            entry = self.chronology_manager.chronology.get_entry_by_id(entry_id)
            if not entry:
                raise HTTPException(status_code=404, detail="Simulation not found")

            if file_type not in entry.result_files:
                raise HTTPException(status_code=404, detail="File not found")

            file_path = Path(entry.result_files[file_type])

            # Security check - ensure file is within output directory
            try:
                full_path = self.output_dir.parent / file_path
                full_path.resolve().relative_to(self.output_dir.parent.resolve())
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied")

            if not full_path.exists():
                raise HTTPException(status_code=404, detail="File not found on disk")

            # Determine media type
            media_type = (
                mimetypes.guess_type(str(full_path))[0] or "application/octet-stream"
            )

            return FileResponse(
                path=str(full_path),
                media_type=media_type,
                filename=f"{entry.simulation_name}_{file_type}_{entry.id}.{full_path.suffix[1:]}",
            )

        return app

    def _generate_download_html(self) -> str:
        """Generate HTML download page."""
        entries = self.chronology_manager.chronology.get_recent_entries(10)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MFC Simulation Downloads</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
                .simulation {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }}
                .success {{ border-left: 5px solid #27ae60; }}
                .failed {{ border-left: 5px solid #e74c3c; }}
                .download-links {{ margin-top: 10px; }}
                .download-links a {{
                    display: inline-block; margin-right: 10px; padding: 8px 16px;
                    background: #3498db; color: white; text-decoration: none; border-radius: 4px;
                }}
                .download-links a:hover {{ background: #2980b9; }}
                .metadata {{ color: #666; font-size: 0.9em; }}
                .tags {{ margin-top: 10px; }}
                .tag {{ background: #ecf0f1; padding: 3px 8px; border-radius: 12px;
                       font-size: 0.8em; margin-right: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 MFC Simulation Downloads</h1>
                <p>Download simulation results, configurations, and data files</p>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """

        if not entries:
            html += """
            <div class="simulation">
                <h3>No simulations found</h3>
                <p>Run some simulations to see download links here.</p>
            </div>
            """
        else:
            for entry in entries:
                status_class = "success" if entry.success else "failed"
                status_icon = "✅" if entry.success else "❌"

                html += f"""
                <div class="simulation {status_class}">
                    <h3>{status_icon} {entry.simulation_name}</h3>
                    <div class="metadata">
                        <strong>ID:</strong> {entry.id} |
                        <strong>Time:</strong> {entry.timestamp} |
                        <strong>Duration:</strong> {entry.duration_hours:.1f}h |
                        <strong>Execution:</strong> {entry.execution_time_seconds:.1f}s
                    </div>

                    {f"<p><strong>Description:</strong> {entry.description}</p>" if entry.description else ""}

                    <div class="tags">
                        {"".join(f'<span class="tag">{tag}</span>' for tag in entry.tags) if entry.tags else ""}
                    </div>
                """

                if entry.result_files:
                    html += '<div class="download-links">'
                    for file_type in entry.result_files:
                        file_label = file_type.replace("_", " ").title()
                        html += f'<a href="/download/{entry.id}/{file_type}" target="_blank">📄 {file_label}</a>'
                    html += "</div>"
                else:
                    html += "<p><em>No download files available</em></p>"

                html += "</div>"

        html += """
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
                <p>MFC Q-Learning Simulation System |
                   <a href="/api/simulations">JSON API</a> |
                   Generated with browser download support</p>
            </div>
        </body>
        </html>
        """

        return html

    def start_server(self, open_browser: bool = True) -> None:
        """Start the download server."""
        if not FASTAPI_AVAILABLE:
            logger.error("Cannot start server - FastAPI not available")
            return

        app = self.create_fastapi_app()
        if not app:
            return

        if open_browser:
            # Open browser after a short delay
            import threading

            def open_browser_delayed() -> None:
                import time

                time.sleep(2)
                webbrowser.open(f"http://localhost:{self.port}")

            threading.Thread(target=open_browser_delayed, daemon=True).start()

        try:
            uvicorn.run(app, host="127.0.0.1", port=self.port, log_level="info")
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.exception(f"Server error: {e}")


def create_streamlit_download_interface() -> None:
    """Create Streamlit interface for downloads (if available)."""
    if not STREAMLIT_AVAILABLE:
        return

    st.title("🔬 MFC Simulation Downloads")
    st.markdown("Download simulation results, configurations, and data files")

    manager = get_chronology_manager()
    entries = manager.chronology.get_recent_entries(20)

    if not entries:
        st.warning("No simulations found. Run some simulations to see downloads here.")
        return

    # Summary metrics
    total_sims = len(manager.chronology.entries)
    successful = len([e for e in manager.chronology.entries if e.success])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Simulations", total_sims)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric(
            "Success Rate",
            f"{(successful / total_sims * 100):.1f}%" if total_sims > 0 else "0%",
        )

    # Filters
    st.sidebar.header("Filters")

    # Tag filter
    all_tags = set()
    for entry in entries:
        all_tags.update(entry.tags)

    selected_tags = st.sidebar.multiselect("Filter by tags", sorted(all_tags))
    show_failed = st.sidebar.checkbox("Show failed simulations", value=True)

    # Filter entries
    filtered_entries = entries
    if selected_tags:
        filtered_entries = [
            e for e in filtered_entries if any(tag in e.tags for tag in selected_tags)
        ]
    if not show_failed:
        filtered_entries = [e for e in filtered_entries if e.success]

    # Display simulations
    for entry in filtered_entries:
        with st.expander(
            f"{'✅' if entry.success else '❌'} {entry.simulation_name} ({entry.id})",
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                if entry.description:
                    st.write(f"**Description:** {entry.description}")
                st.write(f"**Timestamp:** {entry.timestamp}")
                st.write(f"**Duration:** {entry.duration_hours:.1f} hours")
                st.write(
                    f"**Execution Time:** {entry.execution_time_seconds:.1f} seconds",
                )

                if entry.tags:
                    st.write(f"**Tags:** {', '.join(entry.tags)}")

                # Results summary
                if entry.results_summary:
                    st.write("**Key Results:**")
                    for key, value in list(entry.results_summary.items())[:5]:
                        if isinstance(value, int | float):
                            st.write(f"  - {key}: {value:.4f}")
                        else:
                            st.write(f"  - {key}: {value}")

            with col2:
                # Download buttons
                if entry.result_files:
                    st.write("**Downloads:**")
                    for file_type, file_path in entry.result_files.items():
                        file_label = file_type.replace("_", " ").title()
                        full_path = Path(file_path)

                        if full_path.exists():
                            with open(full_path, "rb") as f:
                                st.download_button(
                                    label=f"📄 {file_label}",
                                    data=f.read(),
                                    file_name=f"{entry.simulation_name}_{file_type}_{entry.id}.{full_path.suffix[1:]}",
                                    mime=mimetypes.guess_type(str(full_path))[0],
                                )
                        else:
                            st.error(f"File not found: {file_path}")
                else:
                    st.write("*No download files available*")


def start_download_interface(interface_type: str = "fastapi", port: int = 8080) -> None:
    """Start download interface.

    Args:
        interface_type: Type of interface ('fastapi' or 'streamlit')
        port: Port to run on

    """
    if interface_type == "streamlit" and STREAMLIT_AVAILABLE:
        # For streamlit, we expect it to be run via: streamlit run web_download_server.py
        create_streamlit_download_interface()
    elif interface_type == "fastapi":
        server = DownloadServer(port=port)
        server.start_server()
    else:
        if FASTAPI_AVAILABLE:
            pass
        if STREAMLIT_AVAILABLE:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MFC Simulation Download Server")
    parser.add_argument(
        "--interface",
        choices=["fastapi", "streamlit"],
        default="fastapi",
        help="Interface type to use",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    args = parser.parse_args()

    if args.interface == "fastapi":
        server = DownloadServer(port=args.port)
        server.start_server(open_browser=not args.no_browser)
    else:
        start_download_interface(args.interface, args.port)
