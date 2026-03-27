import typing
from ctypes import c_void_p, CDLL

class CXXwrapper:
    BSQP : CDLL #C/C++ module handle, to be set when this python module is loaded


"""
Holder owning a C/C++ side object via a c_void_p.
May be passed any number of CXXobjHolder holding objects that
the held object relies on being kept alive.
"""
class CXXobjHolder:
    ptr : c_void_p = c_void_p(None)
    deleter : typing.Callable = lambda ptr: None
    deps : typing.List['CXXobjHolder'] = []
    def __init__(self, ptr : c_void_p, deleter : typing.Callable[[c_void_p], None], *deps):
        self.ptr = ptr
        self.deleter = deleter
        self.deps = list(deps)
    def __del__(self):
        self.deleter(self.ptr)

"""
A class that allows creating C/C++ side objects based on its instance,
but does not itself manage the returned C/C++ object. 
"""
class CXXobjCreator(CXXwrapper):
    #To be implemented by child classes
    def create_cxx_obj(self) -> CXXobjHolder:
        raise NotImplementedError(f"create_cxx_obj not implemented for {type(self).__name__}")

"""
A class handling a C/C++ side object and possibly providing
methods to interface with the handled object.
"""
class CXXobjWrapper(CXXwrapper):
    cxx_obj : c_void_p = c_void_p(None)
    def get_cxx_obj(self):
        return self.cxx_obj
    def __del__(self):
        raise NotImplementedError("A destructor for the cxx_obj must be provided")
