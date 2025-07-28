"""
Real-time Monitoring Dashboard Frontend

A modern Streamlit-based dashboard for real-time MFC system monitoring.
Provides comprehensive visualization, control, and alert management.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import json
import requests
import asyncio
import websockets
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import os
import sys

API_BASE_URL = "http://localhost:8000/api"