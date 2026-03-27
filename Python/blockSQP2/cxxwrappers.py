import typing
from ctypes import c_void_p, CDLL


class CXXwrapper:
    """
    Base class for all classes that access the loaded blockSQP2 binary.
    """
    BSQP : CDLL #C/C++ module handle, set in __init__.py

class CXXobjHolder:
    """
    Holder owning a C/C++ side object via a c_void_p.
    May be passed any number of CXXobjHolder holding objects that
    the held object depends on being kept alive.
    """
    ptr : c_void_p = c_void_p(None)
    deleter : typing.Callable = lambda ptr: None
    deps : typing.List['CXXobjHolder'] = []
    def __init__(self, ptr : c_void_p, deleter : typing.Callable[[c_void_p], None], *deps):
        self.ptr = ptr
        self.deleter = deleter
        self.deps = list(deps)
    def __del__(self):
        self.deleter(self.ptr)

class CXXobjCreator(CXXwrapper):
    """
    A class that allows creating C/C++ side objects based on its instance,
    but does not itself manage the returned C/C++ object. 
    """
    #To be implemented in subclasses
    def create_cxx_obj(self) -> CXXobjHolder:
        raise NotImplementedError(f"create_cxx_obj not implemented for {type(self).__name__}")

class CXXobjWrapper(CXXwrapper):
    """
    A class handling a C/C++ side object and possibly providing
    methods to interface with the handled object.
    """
    cxx_obj : c_void_p = c_void_p(None)
    def get_cxx_obj(self):
        return self.cxx_obj
    def __del__(self):
        raise NotImplementedError("Subclasses of CXXobjWrapper must provide a destructor")
