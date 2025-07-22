from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort
from python import Python
from tensor import Tensor, TensorShape
from .mfc_qlearning import MFCQLearningController, MFCQLearningConfig


@export
fn PyInit_qlearning_bindings() -> PythonObject:
    try:
        var m = PythonModuleBuilder("qlearning_bindings")

        # Expose the MFC Q-learning controller
        _ = (
            m.add_type[MFCQLearningController]("MFCQLearningController")
            .def_init_defaultable[MFCQLearningController]()
            .def_method[MFCQLearningController.train]("train")
            .def_method[MFCQLearningController.test_controller](
                "test_controller"
            )
            .def_method[MFCQLearningController.save_q_table]("save_q_table")
        )

        # Expose configuration
        _ = m.add_type[MFCQLearningConfig](
            "MFCQLearningConfig"
        ).def_init_defaultable[MFCQLearningConfig]()

        return m.finalize()

    except e:
        return abort[PythonObject](
            String("failed to create Q-learning bindings module: ", e)
        )


@fieldwise_init
struct PyMFCQLearningController(Copyable, Defaultable, Movable, Representable):
    """Python-compatible wrapper for MFCQLearningController"""

    var controller: MFCQLearningController

    fn __init__(out self):
        var config = MFCQLearningConfig()
        self.controller = MFCQLearningController(config)

    fn __init__(out self, config: MFCQLearningConfig):
        self.controller = MFCQLearningController(config)

    fn __repr__(self) -> String:
        return (
            "PyMFCQLearningController(episodes_trained="
            + str(self.controller.episode_count)
            + ")"
        )

    @staticmethod
    fn __moveinit__(out self: Self, owned existing: Self):
        self.controller = existing.controller^

    @staticmethod
    fn __copyinit__(out self: Self, existing: Self):
        # Deep copy would be complex, so we create a new instance
        var config = MFCQLearningConfig()
        self.controller = MFCQLearningController(config)

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
    fn train_controller(py_self: PythonObject) -> PythonObject:
        """Train the Q-learning controller"""
        var self_ptr = Self._get_self_ptr(py_self)
        self_ptr[].controller.train()
        return PythonObject(True)

    @staticmethod
    fn test_controller(
        py_self: PythonObject, n_episodes: PythonObject
    ) -> PythonObject:
        """Test the trained controller"""
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            var n_test = Int(n_episodes)
            var results = self_ptr[].controller.test_controller(n_test)

            # Convert tensor results to Python list
            var result_list = Python.list()
            var rows = results.shape()[0]
            var cols = results.shape()[1]

            for i in range(rows):
                var row = Python.list()
                for j in range(cols):
                    row.append(results[i, j])
                result_list.append(row)

            return result_list

        except e:
            print("Error in test_controller:", e)
            return PythonObject(None)

    @staticmethod
    fn save_q_table(
        py_self: PythonObject, filename: PythonObject
    ) -> PythonObject:
        """Save Q-table to file"""
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            var fname = String(filename)
            self_ptr[].controller.save_q_table(fname)
            return PythonObject(True)
        except e:
            print("Error saving Q-table:", e)
            return PythonObject(False)

    @staticmethod
    fn get_q_table_stats(py_self: PythonObject) -> PythonObject:
        """Get Q-table statistics"""
        var self_ptr = Self._get_self_ptr(py_self)

        # Calculate basic statistics
        var q_table = self_ptr[].controller.q_table
        var total_elements = q_table.num_elements()

        var total_sum = 0.0
        var min_val = q_table._buffer[0]
        var max_val = q_table._buffer[0]

        for i in range(total_elements):
            var val = q_table._buffer[i]
            total_sum += val
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

        var mean_val = total_sum / total_elements

        # Return as Python dictionary
        var stats = Python.dict()
        stats["shape"] = Python.tuple([q_table.shape()[0], q_table.shape()[1]])
        stats["mean"] = mean_val
        stats["min"] = min_val
        stats["max"] = max_val
        stats["episodes_trained"] = self_ptr[].controller.episode_count

        return stats

    @staticmethod
    fn set_config(
        py_self: PythonObject,
        learning_rate: PythonObject,
        discount_factor: PythonObject,
        epsilon: PythonObject,
    ) -> PythonObject:
        """Update controller configuration"""
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].controller.config.learning_rate = Float64(learning_rate)
            self_ptr[].controller.config.discount_factor = Float64(
                discount_factor
            )
            self_ptr[].controller.config.epsilon = Float64(epsilon)
            self_ptr[].controller.current_epsilon = Float64(epsilon)
            return PythonObject(True)
        except e:
            print("Error setting config:", e)
            return PythonObject(False)


# Update the module builder to include the wrapper
@export
fn PyInit_qlearning_mfc() -> PythonObject:
    try:
        var m = PythonModuleBuilder("qlearning_mfc")

        # Expose the Python-compatible wrapper
        _ = (
            m.add_type[PyMFCQLearningController]("MFCQLearningController")
            .def_init_defaultable[PyMFCQLearningController]()
            .def_method[PyMFCQLearningController.train_controller]("train")
            .def_method[PyMFCQLearningController.test_controller]("test")
            .def_method[PyMFCQLearningController.save_q_table]("save_q_table")
            .def_method[PyMFCQLearningController.get_q_table_stats]("get_stats")
            .def_method[PyMFCQLearningController.set_config]("set_config")
        )

        return m.finalize()

    except e:
        return abort[PythonObject](
            String("failed to create Q-learning MFC module: ", e)
        )
