from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

@export
fn PyInit_test() -> PythonObject:
    try:
        var m = PythonModuleBuilder("test")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))