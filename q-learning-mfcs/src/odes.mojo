from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort
import math
from python import Python


@export
fn PyInit_odes() -> PythonObject:
    try:
        var m = PythonModuleBuilder("odes")
        # Expose the MFCModel struct as a Python class.
        _ = (
            m.add_type[MFCModel]("MFCModel")
            .def_init_defaultable[MFCModel]()
            .def_method[MFCModel.mfc_odes]("mfc_odes")
        )

        return m.finalize()

    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


@fieldwise_init
struct MFCModel(Copyable, Defaultable, Movable, Representable):
    """
    Microbial Fuel Cell (MFC) model parameters and state variables.
    (July 2025 API - Corrected Nested Syntax).
    """

    var F: Float64
    var R: Float64
    var T: Float64
    var k_m: Float64
    var d_m: Float64
    var k_aq: Float64
    var d_cell: Float64
    var C_a: Float64
    var C_c: Float64
    var V_a: Float64
    var V_c: Float64
    var A_m: Float64
    var Y_ac: Float64
    var K_dec: Float64
    var f_x: Float64
    var alpha: Float64
    var beta: Float64
    var k1_0: Float64
    var k2_0: Float64
    var K_AC: Float64
    var K_O2: Float64
    var Q_a: Float64
    var Q_c: Float64
    var C_AC_in: Float64
    var C_CO2_in: Float64
    var X_in: Float64
    var C_H_in: Float64
    var C_O2_in: Float64
    var C_M_in: Float64
    var C_OH_in: Float64
    var U0: Float64
    var C_AC: Float64
    var C_CO2: Float64
    var C_H: Float64
    var X: Float64
    var C_O2: Float64
    var C_OH: Float64
    var C_M: Float64
    var eta_a: Float64
    var eta_c: Float64

    fn __init__(out self):
        """Initializes with default parameters using correct Mojo syntax."""
        self.F = 96485.332
        self.R = 8.314
        self.T = 303
        self.k_m = 17.0
        self.d_m = 1.778e-4
        self.k_aq = 5.0
        self.d_cell = 2.2e-2
        self.C_a = 4e2
        self.C_c = 5e2
        self.V_a = 5.5e-5
        self.V_c = 5.5e-5
        self.A_m = 5.0e-4
        self.Y_ac = 0.05
        self.K_dec = 8.33e-4
        self.f_x = 10.0
        self.alpha = 0.051
        self.beta = 0.063
        self.k1_0 = 0.207
        self.k2_0 = 3.288e-5
        self.K_AC = 0.592
        self.K_O2 = 0.004
        self.Q_a = 2.25e-5
        self.Q_c = 1.11e-3
        self.C_AC_in = 1.56
        self.C_CO2_in = 0.0
        self.X_in = 0.0
        self.C_H_in = 0.0
        self.C_O2_in = 0.3125
        self.C_M_in = 0.0
        self.C_OH_in = 0.0
        self.U0 = 0.77
        self.C_AC = 0.0
        self.C_CO2 = 0.0
        self.C_H = 0.0
        self.X = 0.0
        self.C_O2 = 0.0
        self.C_OH = 0.0
        self.C_M = 0.0
        self.eta_a = 0.0
        self.eta_c = 0.0

    fn __repr__(self: MFCModel) -> String:
        var repr = (
            "MFCModel("
            + "F={self.F}, R={self.R}, T={self.T}, k_m={self.k_m},"
            " d_m={self.d_m}, "
            + "k_aq={self.k_aq}, d_cell={self.d_cell}, C_a={self.C_a},"
            " C_c={self.C_c}, "
            + "V_a={self.V_a}, V_c={self.V_c}, A_m={self.A_m},"
            " Y_ac={self.Y_ac}, "
            + "K_dec={self.K_dec}, f_x={self.f_x}, alpha={self.alpha},"
            " beta={self.beta}, "
            + "k1_0={self.k1_0}, k2_0={self.k2_0}, K_AC={self.K_AC},"
            " K_O2={self.K_O2}, "
            + "Q_a={self.Q_a}, Q_c={self.Q_c}, C_AC_in={self.C_AC_in},"
            " C_CO2_in={self.C_CO2_in}, "
            + "X_in={self.X_in}, C_H_in={self.C_H_in}, C_O2_in={self.C_O2_in}, "
            + "C_M_in={self.C_M_in}, C_OH_in={self.C_OH_in}, U0={self.U0}, "
            + "C_AC={self.C_AC}, C_CO2={self.C_CO2}, C_H={self.C_H},"
            " X={self.X}, "
            + "C_O2={self.C_O2}, C_OH={self.C_OH}, C_M={self.C_M}, "
            + "eta_a={self.eta_a}, eta_c={self.eta_c})"
        )
        return repr

    @staticmethod
    fn __moveinit__(out self: Self, owned existing: Self):
        """
        Move initializer for MFCModel.
        This is called when an instance is moved, ensuring proper ownership transfer.
        """
        self.F = existing.F
        self.R = existing.R
        self.T = existing.T
        self.k_m = existing.k_m
        self.d_m = existing.d_m
        self.k_aq = existing.k_aq
        self.d_cell = existing.d_cell
        self.C_a = existing.C_a
        self.C_c = existing.C_c
        self.V_a = existing.V_a
        self.V_c = existing.V_c
        self.A_m = existing.A_m
        self.Y_ac = existing.Y_ac
        self.K_dec = existing.K_dec
        self.f_x = existing.f_x
        self.alpha = existing.alpha
        self.beta = existing.beta
        self.k1_0 = existing.k1_0
        self.k2_0 = existing.k2_0
        self.K_AC = existing.K_AC
        self.K_O2 = existing.K_O2
        self.Q_a = existing.Q_a
        self.Q_c = existing.Q_c
        self.C_AC_in = existing.C_AC_in
        self.C_CO2_in = existing.C_CO2_in
        self.X_in = existing.X_in
        self.C_H_in = existing.C_H_in
        self.C_O2_in = existing.C_O2_in
        self.C_M_in = existing.C_M_in
        self.C_OH_in = existing.C_OH_in
        self.U0 = existing.U0
        self.C_AC = 0.0  # Reset to default value on move init.
        self.C_CO2 = 0.0  # Reset to default value on move init.
        self.C_H = 0.0  # Reset to default value on move init.
        self.X = 0.0  # Reset to default value on move init.
        self.C_O2 = 0.0  # Reset to default value on move init.
        self.C_OH = 0.0  # Reset to default value on move init.
        self.C_M = 0.0  # Reset to default value on move init.
        self.eta_a = 0.0  # Reset to default value on move init.
        self.eta_c = 0.0  # Reset to default value on move init.

    @staticmethod
    fn __copyinit__(out self: Self, existing: Self):
        """
        Copy initializer for MFCModel.
        This is called when an instance is copied, ensuring proper value transfer.
        """
        self.F = existing.F
        self.R = existing.R
        self.T = existing.T
        self.k_m = existing.k_m
        self.d_m = existing.d_m
        self.k_aq = existing.k_aq
        self.d_cell = existing.d_cell
        self.C_a = existing.C_a
        self.C_c = existing.C_c
        self.V_a = existing.V_a
        self.V_c = existing.V_c
        self.A_m = existing.A_m
        self.Y_ac = existing.Y_ac
        self.K_dec = existing.K_dec
        self.f_x = existing.f_x
        self.alpha = existing.alpha
        self.beta = existing.beta
        self.k1_0 = existing.k1_0
        self.k2_0 = existing.k2_0
        self.K_AC = existing.K_AC
        self.K_O2 = existing.K_O2
        self.Q_a = existing.Q_a
        self.Q_c = existing.Q_c
        self.C_AC_in = existing.C_AC_in
        self.C_CO2_in = existing.C_CO2_in
        self.X_in = existing.X_in
        self.C_H_in = existing.C_H_in
        self.C_O2_in = existing.C_O2_in
        self.C_M_in = existing.C_M_in
        self.C_OH_in = existing.C_OH_in
        self.U0 = existing.U0
        self.C_AC = existing.C_AC
        self.C_CO2 = existing.C_CO2
        self.C_H = existing.C_H
        self.X = existing.X
        self.C_O2 = existing.C_O2
        self.C_OH = existing.C_OH
        self.C_M = existing.C_M
        self.eta_a = existing.eta_a
        self.eta_c = existing.eta_c

    @staticmethod
    fn _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            return abort[UnsafePointer[Self]](
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type: "
                    ),
                    e,
                )
            )

    @staticmethod
    fn mfc_odes(
        py_self: PythonObject,
        t: PythonObject,
        y: PythonObject,
        i_fc: PythonObject,
    ) raises -> PythonObject:
        """
        Calculates the derivatives. This now accepts a NumPy array directly
        and returns a Mojo List, which is automatically converted for Python.
        """
        # Add a check for the input array's length for robustness.
        var self_ptr = Self._get_self_ptr(py_self)

        try:
            if y._len != 9:
                # This error message will be propagated to Python as an exception.
                raise Error("Input list 'y' must have exactly 9 elements.")
        except:
            PythonObject()

        var C_AC = Float64(y[0])
        var C_CO2 = Float64(y[1])
        var C_H = Float64(y[2])
        var X = Float64(y[3])
        var C_O2 = Float64(y[4])
        var C_OH = Float64(y[5])
        var C_M = Float64(y[6])
        var eta_a = Float64(y[7])
        var eta_c = Float64(y[8])

        var r1: Float64 = (
            self_ptr[].k1_0
            * math.exp(
                (self_ptr[].alpha * self_ptr[].F)
                / (self_ptr[].R * self_ptr[].T)
                * eta_a
            )
            * (C_AC / (self_ptr[].K_AC + C_AC))
            * X
        )
        var r2: Float64 = (
            -self_ptr[].k2_0
            * (C_O2 / (self_ptr[].K_O2 + C_O2))
            * math.exp(
                (self_ptr[].beta - 1.0)
                * self_ptr[].F
                / (self_ptr[].R * self_ptr[].T)
                * eta_c
            )
        )
        var N_M: Float64 = (3600.0 * Float64(i_fc)) / self_ptr[].F

        var dC_AC_dt: Float64 = (
            self_ptr[].Q_a * (self_ptr[].C_AC_in - C_AC) - self_ptr[].A_m * r1
        ) / self_ptr[].V_a
        var dC_CO2_dt: Float64 = (
            self_ptr[].Q_a * (self_ptr[].C_CO2_in - C_CO2)
            + 2.0 * self_ptr[].A_m * r1
        ) / self_ptr[].V_a
        var dC_H_dt: Float64 = (
            self_ptr[].Q_a * (self_ptr[].C_H_in - C_H)
            + 8.0 * self_ptr[].A_m * r1
        ) / self_ptr[].V_a
        var dX_dt: Float64 = (
            self_ptr[].Q_a * (self_ptr[].X_in - X) / self_ptr[].f_x
            + self_ptr[].A_m * self_ptr[].Y_ac * r1
        ) / self_ptr[].V_a - self_ptr[].K_dec * X

        var dC_O2_dt: Float64 = (
            self_ptr[].Q_c * (self_ptr[].C_O2_in - C_O2) + r2 * self_ptr[].A_m
        ) / self_ptr[].V_c
        var dC_OH_dt: Float64 = (
            self_ptr[].Q_c * (self_ptr[].C_OH_in - C_OH)
            - 4.0 * r2 * self_ptr[].A_m
        ) / self_ptr[].V_c
        var dC_M_dt: Float64 = (
            self_ptr[].Q_c * (self_ptr[].C_M_in - C_M) + N_M * self_ptr[].A_m
        ) / self_ptr[].V_c

        var d_eta_a_dt: Float64 = (
            3600.0 * Float64(i_fc) - 8.0 * self_ptr[].F * r1
        ) / self_ptr[].C_a
        var d_eta_c_dt: Float64 = (
            -3600.0 * Float64(i_fc) - 4.0 * self_ptr[].F * r2
        ) / self_ptr[].C_c

        var derivatives: List[Float64] = [
            dC_AC_dt,
            dC_CO2_dt,
            dC_H_dt,
            dX_dt,
            dC_O2_dt,
            dC_OH_dt,
            dC_M_dt,
            d_eta_a_dt,
            d_eta_c_dt,
        ]

        return Python.list(derivatives)
