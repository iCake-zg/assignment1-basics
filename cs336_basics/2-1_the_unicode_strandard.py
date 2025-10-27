


'''
Qa: chr(0) 的结果为'\x00'
'''
print("2-1 Q1:")
print(chr(0))


'''
Qb: __repr__() 会返回带引号的官方字符串表示
'''
s = 'sss'
print(s)
print(s.__repr__())


'''
Qc:
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
'''
