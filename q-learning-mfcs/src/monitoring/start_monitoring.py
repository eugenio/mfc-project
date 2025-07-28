"""
MFC Real-time Monitoring System Startup Script

Starts all monitoring system components:
- Dashboard API server
- Real-time data streaming service  
- Safety monitoring system
- Frontend dashboard
"""
import asyncio
import subprocess
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp

MONITORING_DIR = Path(__file__).parent
SRC_DIR = MONITORING_DIR.parent
PROJECT_DIR = SRC_DIR.parent
class MonitoringSystemManager:
    """Manager for MFC monitoring system components"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.is_running = False
        
    def start_dashboard_api(self) -> subprocess.Popen:
        """Start the dashboard API server"""
        logger.info("Starting Dashboard API server...")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "monitoring.dashboard_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=SRC_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return process
    
    def start_realtime_streamer(self) -> subprocess.Popen:
        """Start the real-time streaming service"""
        logger.info("Starting Real-time Streaming service...")
        
        cmd = [
            sys.executable, "-m",
            "monitoring.realtime_streamer"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=SRC_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return process
    
    def start_frontend_dashboard(self) -> subprocess.Popen:
        """Start the Streamlit frontend dashboard"""
        logger.info("Starting Frontend Dashboard...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(MONITORING_DIR / "dashboard_frontend.py"),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=SRC_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return process
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "plotly",
            "websockets", "pandas", "numpy"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        import requests
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        return False
    
    def start_all_services(self):
        """Start all monitoring services"""
        if not self.check_dependencies():
            logger.error("Cannot start services due to missing dependencies")
            return False
        
        try:
            # Start Dashboard API
            self.processes["dashboard_api"] = self.start_dashboard_api()
            time.sleep(2)  # Give it time to start
            
            # Start Real-time Streamer
            self.processes["realtime_streamer"] = self.start_realtime_streamer()
            time.sleep(2)
            
            # Start Frontend Dashboard
            self.processes["frontend_dashboard"] = self.start_frontend_dashboard()
            time.sleep(3)
            
            self.is_running = True
            
            # Check if services are running
            if not self.wait_for_service("http://localhost:8000/api/health"):
                logger.warning("Dashboard API may not have started properly")
            else:
                logger.info("✅ Dashboard API is running on http://localhost:8000")
            
            if not self.wait_for_service("http://localhost:8501"):
                logger.warning("Frontend Dashboard may not have started properly")
            else:
                logger.info("✅ Frontend Dashboard is running on http://localhost:8501")
            
            logger.info("✅ Real-time Streamer is running on ws://localhost:8001")
            
            logger.info("\n🎉 MFC Monitoring System is fully operational!")
            logger.info("\n📊 Access points:")
            logger.info("   • Dashboard UI: http://localhost:8501")
            logger.info("   • API Documentation: http://localhost:8000/api/docs")
            logger.info("   • WebSocket Stream: ws://localhost:8001/ws")
            logger.info("   • Health Check: http://localhost:8000/api/health")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            self.stop_all_services()
            return False
    
    def stop_all_services(self):
        """Stop all monitoring services"""
        logger.info("Stopping all monitoring services...")
        
        for service_name, process in self.processes.items():
            try:
                logger.info(f"Stopping {service_name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {service_name}...")
                    process.kill()
                    process.wait()
                
                logger.info(f"✅ {service_name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")
        
        self.processes.clear()
        self.is_running = False
        logger.info("🛑 All services stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            "is_running": self.is_running,
            "services": {}
        }
        
        for service_name, process in self.processes.items():
            if process.poll() is None:
                status["services"][service_name] = "running"
            else:
                status["services"][service_name] = "stopped"
        
        return status
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        if service_name not in self.processes:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        logger.info(f"Restarting {service_name}...")
        
        # Stop the service
        process = self.processes[service_name]
        process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        # Start the service again
        try:
            if service_name == "dashboard_api":
                self.processes[service_name] = self.start_dashboard_api()
            elif service_name == "realtime_streamer":
                self.processes[service_name] = self.start_realtime_streamer()
            elif service_name == "frontend_dashboard":
                self.processes[service_name] = self.start_frontend_dashboard()
            
            logger.info(f"✅ {service_name} restarted")
            return True
            
        except Exception as e:
            logger.error(f"Error restarting {service_name}: {e}")
            return False
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if 'manager' in globals():
        manager.stop_all_services()
    sys.exit(0)
def main():
    """Main function"""
    global manager
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create manager
    manager = MonitoringSystemManager()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            success = manager.start_all_services()
            if success:
                # Keep running until interrupted
                try:
                    while manager.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    manager.stop_all_services()
            else:
                sys.exit(1)
        
        elif command == "stop":
            manager.stop_all_services()
        
        elif command == "status":
            status = manager.get_status()
            print(f"System running: {status['is_running']}")
            for service, state in status['services'].items():
                print(f"  {service}: {state}")
        
        elif command == "restart":
            if len(sys.argv) > 2:
                service_name = sys.argv[2]
                manager.restart_service(service_name)
            else:
                manager.stop_all_services()
                time.sleep(2)
                manager.start_all_services()
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python start_monitoring.py [start|stop|status|restart [service_name]]")
            sys.exit(1)
    
    else:
        # Default: start all services
        success = manager.start_all_services()
        if success:
            try:
                while manager.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                manager.stop_all_services()
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()