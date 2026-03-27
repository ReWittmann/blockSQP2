from ctypes import c_char_p
from .cxxwrappers import CXXobjWrapper

class Stats(CXXobjWrapper):
    def __init__(self, outpath : str):
        self.cxx_obj = self.BSQP.create_SQPstats(c_char_p(outpath.encode('utf-8')))
        
    def __del__(self):
        self.BSQP.delete_SQPstats(self.cxx_obj)
    @property
    def itCount(self):
        return self.BSQP.SQPstats_get_itCount(self.cxx_obj)