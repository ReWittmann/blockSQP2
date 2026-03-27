import typing
from ctypes import c_void_p, CDLL

#Classes meant to be passed to C/C++ code.
# Must implement create_cxx_obj, a method constructing the corresponding
# C/C++ class instance and returning a c_void_p to it, and a cxx_obj_deleter,
# the deleter method for the instance taking the c_void_p.
class CXXargument:
    BSQP : CDLL
    
    def create_cxx_obj(self):
        raise NotImplementedError(f"create_cxx_obj not implemented for {type(self).__name__}")
    @staticmethod
    def cxx_obj_deleter(cxx_obj):
        raise NotImplementedError("Missing implementation of the cxx_obj_deleter method in this CXXwrapper subclass")
    
    @classmethod
    def delete_cxx_obj(cls, cxx_obj):
        cls.cxx_obj_deleter(cxx_obj)

# Holder for C/C++ objects with automatic deletion
class CXXholder:
    ptr : c_void_p
    deleter : typing.Callable
    def __init__(self, ptr : c_void_p, deleter : typing.Callable[[c_void_p], None]):
        self.ptr = ptr
        self.deleter = deleter
    def __del__(self):
        self.deleter(self.ptr)

