from ctypes import c_void_p, c_char_p
class Stats:
    BSQP = None
    SQPstats_obj : c_void_p = c_void_p(None)
    def __init__(self, outpath : str):
        self.SQPstats_obj = self.BSQP.create_SQPstats(c_char_p(outpath.encode('utf-8')))
        
    def __del__(self):
        self.BSQP.delete_SQPstats(self.SQPstats_obj)
    
    @property
    def itCount(self):
        return self.BSQP.SQPstats_get_itCount(self.SQPstats_obj)