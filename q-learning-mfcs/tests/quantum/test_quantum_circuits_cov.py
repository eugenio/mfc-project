"""Tests for quantum/quantum_circuits.py - coverage target 98%+."""
import sys
import os
import json
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quantum.quantum_circuits import (
    CircuitValidator,
    QuantumCircuit,
    QuantumGate,
    QuantumGateType,
    QuantumSimulator,
)


class TestQuantumGateType:
    def test_all_gate_types(self):
        assert QuantumGateType.HADAMARD.value == "H"
        assert QuantumGateType.PAULI_X.value == "X"
        assert QuantumGateType.PAULI_Y.value == "Y"
        assert QuantumGateType.PAULI_Z.value == "Z"
        assert QuantumGateType.CNOT.value == "CNOT"
        assert QuantumGateType.RZ.value == "RZ"
        assert QuantumGateType.RY.value == "RY"
        assert QuantumGateType.RX.value == "RX"
        assert QuantumGateType.TOFFOLI.value == "TOFFOLI"
        assert QuantumGateType.SWAP.value == "SWAP"


class TestQuantumGate:
    def test_basic_gate(self):
        gate = QuantumGate(QuantumGateType.HADAMARD, [0])
        assert gate.gate_type == QuantumGateType.HADAMARD
        assert gate.target_qubits == [0]
        assert gate.control_qubits is None
        assert gate.parameters is None

    def test_rotation_gate_valid(self):
        gate = QuantumGate(QuantumGateType.RX, [0], parameters=[1.57])
        assert gate.parameters == [1.57]

    def test_rotation_gate_missing_params(self):
        with pytest.raises(ValueError, match="requires exactly one parameter"):
            QuantumGate(QuantumGateType.RX, [0])

    def test_rotation_gate_wrong_param_count(self):
        with pytest.raises(ValueError, match="requires exactly one parameter"):
            QuantumGate(QuantumGateType.RY, [0], parameters=[1.0, 2.0])

    def test_cnot_valid(self):
        gate = QuantumGate(QuantumGateType.CNOT, [1], control_qubits=[0])
        assert gate.control_qubits == [0]

    def test_cnot_no_control(self):
        with pytest.raises(ValueError, match="requires exactly one control"):
            QuantumGate(QuantumGateType.CNOT, [1])

    def test_cnot_multiple_targets(self):
        with pytest.raises(ValueError, match="requires exactly one target"):
            QuantumGate(QuantumGateType.CNOT, [0, 1], control_qubits=[2])

    def test_rz_gate(self):
        gate = QuantumGate(QuantumGateType.RZ, [0], parameters=[3.14])
        assert gate.parameters == [3.14]


class TestQuantumCircuit:
    def test_init(self):
        qc = QuantumCircuit(3, "test")
        assert qc.num_qubits == 3
        assert qc.name == "test"
        assert len(qc.gates) == 0
        assert qc.depth == 0

    def test_init_zero_qubits(self):
        with pytest.raises(ValueError, match="must be positive"):
            QuantumCircuit(0)

    def test_init_negative_qubits(self):
        with pytest.raises(ValueError, match="must be positive"):
            QuantumCircuit(-1)

    def test_add_gate(self):
        qc = QuantumCircuit(2)
        gate = QuantumGate(QuantumGateType.HADAMARD, [0])
        qc.add_gate(gate)
        assert len(qc.gates) == 1
        assert qc.depth == 1

    def test_add_gate_invalid_qubit(self):
        qc = QuantumCircuit(2)
        gate = QuantumGate(QuantumGateType.HADAMARD, [5])
        with pytest.raises(ValueError, match="Invalid qubit"):
            qc.add_gate(gate)

    def test_add_gate_negative_qubit(self):
        qc = QuantumCircuit(2)
        gate = QuantumGate(QuantumGateType.HADAMARD, [-1])
        with pytest.raises(ValueError, match="Invalid qubit"):
            qc.add_gate(gate)

    def test_hadamard(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        assert qc.gates[0].gate_type == QuantumGateType.HADAMARD

    def test_pauli_x(self):
        qc = QuantumCircuit(2)
        qc.pauli_x(1)
        assert qc.gates[0].gate_type == QuantumGateType.PAULI_X

    def test_pauli_y(self):
        qc = QuantumCircuit(2)
        qc.pauli_y(0)
        assert qc.gates[0].gate_type == QuantumGateType.PAULI_Y

    def test_pauli_z(self):
        qc = QuantumCircuit(2)
        qc.pauli_z(0)
        assert qc.gates[0].gate_type == QuantumGateType.PAULI_Z

    def test_rx(self):
        qc = QuantumCircuit(2)
        qc.rx(0, 1.57)
        assert qc.gates[0].gate_type == QuantumGateType.RX
        assert qc.gates[0].parameters == [1.57]

    def test_ry(self):
        qc = QuantumCircuit(2)
        qc.ry(1, 0.5)
        assert qc.gates[0].gate_type == QuantumGateType.RY

    def test_rz(self):
        qc = QuantumCircuit(2)
        qc.rz(0, 3.14)
        assert qc.gates[0].gate_type == QuantumGateType.RZ

    def test_cnot(self):
        qc = QuantumCircuit(2)
        qc.cnot(0, 1)
        assert qc.gates[0].gate_type == QuantumGateType.CNOT
        assert qc.gates[0].control_qubits == [0]
        assert qc.gates[0].target_qubits == [1]

    def test_toffoli(self):
        qc = QuantumCircuit(3)
        qc.toffoli(0, 1, 2)
        assert qc.gates[0].gate_type == QuantumGateType.TOFFOLI

    def test_swap(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        assert qc.gates[0].gate_type == QuantumGateType.SWAP

    def test_measure(self):
        qc = QuantumCircuit(2)
        qc.measure(0)
        assert 0 in qc.measurements

    def test_measure_duplicate(self):
        qc = QuantumCircuit(2)
        qc.measure(0)
        qc.measure(0)
        assert qc.measurements.count(0) == 1

    def test_measure_invalid(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="Invalid qubit"):
            qc.measure(5)

    def test_measure_negative(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="Invalid qubit"):
            qc.measure(-1)

    def test_measure_all(self):
        qc = QuantumCircuit(3)
        qc.measure_all()
        assert len(qc.measurements) == 3

    def test_depth_property(self):
        qc = QuantumCircuit(2)
        assert qc.depth == 0
        qc.hadamard(0)
        assert qc.depth == 1

    def test_get_gate_count(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        qc.hadamard(1)
        qc.cnot(0, 1)
        counts = qc.get_gate_count()
        assert counts["H"] == 2
        assert counts["CNOT"] == 1

    def test_create_qaoa_circuit(self):
        qc = QuantumCircuit(3)
        qc.create_qaoa_circuit([0.5], [0.3], p=1)
        assert len(qc.gates) > 0

    def test_create_qaoa_circuit_wrong_params(self):
        qc = QuantumCircuit(3)
        with pytest.raises(ValueError, match="Parameter lengths"):
            qc.create_qaoa_circuit([0.5, 0.3], [0.3], p=1)

    def test_create_qaoa_circuit_multi_layer(self):
        qc = QuantumCircuit(2)
        qc.create_qaoa_circuit([0.5, 0.3], [0.2, 0.4], p=2)
        assert len(qc.gates) > 0

    def test_create_vqe_ansatz(self):
        qc = QuantumCircuit(2)
        params = [0.1, 0.2, 0.3, 0.4]
        qc.create_vqe_ansatz(params)
        assert len(qc.gates) > 0

    def test_create_vqe_ansatz_wrong_params(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError, match="Expected 4 parameters"):
            qc.create_vqe_ansatz([0.1])

    def test_to_dict(self):
        qc = QuantumCircuit(2, "test_circuit")
        qc.hadamard(0)
        qc.cnot(0, 1)
        qc.measure_all()
        d = qc.to_dict()
        assert d["name"] == "test_circuit"
        assert d["num_qubits"] == 2
        assert len(d["gates"]) == 2
        assert len(d["measurements"]) == 2

    def test_to_json(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        j = qc.to_json()
        data = json.loads(j)
        assert data["num_qubits"] == 2


class TestCircuitValidator:
    def test_validate_empty_circuit(self):
        qc = QuantumCircuit(2)
        report = CircuitValidator.validate_circuit(qc)
        assert report["valid"] is True
        assert "Circuit contains no gates" in report["warnings"]

    def test_validate_circuit_with_gates(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        qc.cnot(0, 1)
        qc.measure_all()
        report = CircuitValidator.validate_circuit(qc)
        assert report["valid"] is True
        assert report["metrics"]["total_gates"] == 2

    def test_validate_circuit_high_depth(self):
        qc = QuantumCircuit(2)
        for _ in range(60):
            qc.hadamard(0)
        report = CircuitValidator.validate_circuit(qc)
        assert any("depth is high" in w for w in report["warnings"])

    def test_validate_circuit_many_two_qubit_gates(self):
        qc = QuantumCircuit(2)
        for _ in range(15):
            qc.cnot(0, 1)
        report = CircuitValidator.validate_circuit(qc)
        assert any("two-qubit gates" in w for w in report["warnings"])

    def test_validate_gate_duplicate_qubits(self):
        gate = QuantumGate(QuantumGateType.SWAP, [0, 0])
        with pytest.raises(ValueError, match="duplicate qubit"):
            CircuitValidator._validate_gate(gate, 3)

    def test_validate_gate_out_of_bounds(self):
        gate = QuantumGate(QuantumGateType.HADAMARD, [5])
        with pytest.raises(ValueError, match="out of bounds"):
            CircuitValidator._validate_gate(gate, 3)

    def test_count_two_qubit_gates(self):
        qc = QuantumCircuit(3)
        qc.cnot(0, 1)
        qc.swap(1, 2)
        qc.hadamard(0)
        count = CircuitValidator._count_two_qubit_gates(qc)
        assert count == 2

    def test_optimize_circuit_removes_canceling_gates(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        qc.hadamard(0)  # cancels
        qc.pauli_x(1)
        optimized = CircuitValidator.optimize_circuit(qc)
        assert len(optimized.gates) == 1  # only pauli_x remains

    def test_optimize_circuit_no_change(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        qc.pauli_x(1)
        optimized = CircuitValidator.optimize_circuit(qc)
        assert len(optimized.gates) == 2

    def test_optimize_circuit_copies_measurements(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        qc.measure_all()
        optimized = CircuitValidator.optimize_circuit(qc)
        assert len(optimized.measurements) == 2

    def test_gates_cancel_same_type(self):
        g1 = QuantumGate(QuantumGateType.PAULI_X, [0])
        g2 = QuantumGate(QuantumGateType.PAULI_X, [0])
        assert CircuitValidator._gates_cancel(g1, g2) is True

    def test_gates_cancel_different_type(self):
        g1 = QuantumGate(QuantumGateType.PAULI_X, [0])
        g2 = QuantumGate(QuantumGateType.PAULI_Y, [0])
        assert CircuitValidator._gates_cancel(g1, g2) is False

    def test_gates_cancel_different_qubits(self):
        g1 = QuantumGate(QuantumGateType.PAULI_X, [0])
        g2 = QuantumGate(QuantumGateType.PAULI_X, [1])
        assert CircuitValidator._gates_cancel(g1, g2) is False

    def test_gates_cancel_rotation_not_self_inverse(self):
        g1 = QuantumGate(QuantumGateType.RX, [0], parameters=[1.0])
        g2 = QuantumGate(QuantumGateType.RX, [0], parameters=[1.0])
        assert CircuitValidator._gates_cancel(g1, g2) is False

    def test_gates_cancel_swap(self):
        g1 = QuantumGate(QuantumGateType.SWAP, [0, 1])
        g2 = QuantumGate(QuantumGateType.SWAP, [0, 1])
        assert CircuitValidator._gates_cancel(g1, g2) is True


class TestQuantumSimulator:
    def test_init(self):
        sim = QuantumSimulator(shots=512)
        assert sim.shots == 512
        assert sim.noise_model is None

    def test_run_simple_circuit(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        qc.measure_all()
        sim = QuantumSimulator(shots=100)
        results = sim.run(qc)
        assert isinstance(results, dict)
        assert sum(results.values()) == 100

    def test_run_no_measurements(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        sim = QuantumSimulator()
        with pytest.raises(ValueError, match="must have measurements"):
            sim.run(qc)

    def test_run_complex_circuit(self):
        qc = QuantumCircuit(3)
        for _ in range(15):
            qc.hadamard(0)
        qc.measure_all()
        sim = QuantumSimulator(shots=200)
        results = sim.run(qc)
        assert sum(results.values()) == 200

    def test_get_statevector(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        sim = QuantumSimulator()
        sv = sim.get_statevector(qc)
        assert sv.shape == (4,)
        assert sv.dtype == np.complex128
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-10

    def test_get_expectation_value(self):
        qc = QuantumCircuit(2)
        qc.hadamard(0)
        sim = QuantumSimulator()
        ev = sim.get_expectation_value(qc, "Z")
        assert -1 <= ev <= 1


class TestQuantumInit:
    def test_quantum_init_imports(self):
        from quantum import CircuitValidator, QuantumCircuit, QuantumGate
        assert QuantumCircuit is not None
        assert QuantumGate is not None
        assert CircuitValidator is not None

    def test_quantum_version(self):
        import quantum
        assert quantum.__version__ == "1.0.0"

    def test_quantum_all(self):
        import quantum
        assert "QuantumCircuit" in quantum.__all__
        assert "QuantumGate" in quantum.__all__
        assert "CircuitValidator" in quantum.__all__
