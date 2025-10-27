


'''
Qa: UTF8 在 UTF16和32之间，UTF8 对ASCII 字符仅用1个字节编码，而 UTF16 和 UTF32 对所有字符都用2个或4个字节编码。
'''


'''
Qb: 在中文标签下 unicode 会出现单个编码无法解码的问题
'''


def decode_utf8_bytes_to_str_wrong(bytestring:bytes):

    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


# if __name__ == "__main__":
#     print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
#     print(decode_utf8_bytes_to_str_wrong("test".encode("utf-8")))



'''
Qc:
'''

