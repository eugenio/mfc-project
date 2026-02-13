# MFC Data Architecture Optimization Plan

**CRITICAL ARCHITECTURE DOCUMENT**
==================================
This document describes the implemented 3-phase optimization for MFC simulation data architecture.

**MODIFICATION WARNING:** Any changes to the implemented phases must:
1. Preserve the existing queue-based streaming architecture (Phase 1)
2. Maintain incremental update mechanisms for performance (Phase 2)  
3. Follow the planned Parquet migration strategy (Phase 3)
4. Respect existing method signatures and data flow patterns
5. Test thoroughly as this affects real-time simulation performance

**Integration Requirements:**
- All new features must work with the SimulationRunner class structure
- Data flow must remain: simulation -> queue -> incremental updates -> GUI
- Performance optimizations must not break backward compatibility
- Changes must be validated with test scripts before deployment

## Phase 1: Shared Memory Queue Optimization ✅ COMPLETED

## Implementation Plan

### Goal
Replace file-based I/O with in-memory queue system for real-time data streaming between simulation and GUI threads.

### Architecture Changes

#### 1. Data Queues
- **Status Queue**: `queue.Queue()` - simulation status/control messages
- **Data Queue**: `queue.Queue(maxsize=100)` - real-time simulation data points  
- **Buffer**: `live_data_buffer[]` - GUI-side rolling buffer (1000 points max)

#### 2. Data Flow
```
Simulation Thread → data_queue.put(data_point) → GUI reads via get_live_data()
                ↓
              CSV.gz (async backup only)
```

#### 3. Implementation Steps

1. **Add Data Infrastructure**
   - Add `data_queue` and `live_data_buffer` to `SimulationRunner.__init__()`
   - Add `get_live_data()` method for non-blocking data retrieval
   - Add `get_buffered_data()` method to return DataFrame from buffer

2. **Modify Simulation Loop**
   - Send data points to queue instead of immediate file write
   - Maintain CSV.gz as async backup (reduced frequency)
   - Add data point structure: `{timestamp, step, metrics...}`

3. **Update GUI Data Loading**
   - Replace `load_simulation_data()` calls with `get_live_data()`
   - Implement incremental buffer updates
   - Fallback to file loading for historical data

4. **Performance Optimizations**
   - Queue maxsize=100 to prevent memory buildup
   - Rolling buffer (1000 points) to limit GUI memory
   - Async file backup every 100 steps instead of 30

### Expected Performance Gains
- **80% reduction** in I/O overhead during simulation
- **Real-time responsiveness** - no file read delays
- **Minimal memory usage** - bounded queues and buffers
- **Backward compatibility** - CSV.gz backup maintained

### Risk Mitigation
- Queue size limits prevent memory leaks
- File backup ensures data persistence 
- Non-blocking reads prevent GUI freezing
- Graceful fallback to file-based loading

### Testing Strategy
1. Unit tests for queue operations
2. Performance benchmarks (before/after)
3. Memory usage monitoring
4. Data integrity validation (queue vs file)
5. GUI responsiveness testing

### Timeline
- **Setup**: 1 hour (queues, methods)
- **Simulation Integration**: 1-2 hours  
- **GUI Integration**: 1-2 hours
- **Testing/Validation**: 1 hour
- **Total**: 4-6 hours