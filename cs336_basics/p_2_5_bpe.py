



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from multiprocessing import Pool
    import regex as re
    from typing import Union,BinaryIO
    from collections import defaultdict,Counter
    import os
    from cs336_basics.pretokenization_example import find_chunk_boundaries
    base_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    initialize_vocab = {1: b'!', 2: b'"', 3: b'#', 4: b'$', 5: b'%', 6: b'&', 7: b"'", 8: b'(', 9: b')', 10: b'*', 11: b'+', 12: b',', 13: b'-', 14: b'.', 15: b'/', 16: b'0', 17: b'1', 18: b'2', 19: b'3', 20: b'4', 21: b'5', 22: b'6', 23: b'7', 24: b'8', 25: b'9', 26: b':', 27: b';', 28: b'<', 29: b'=', 30: b'>', 31: b'?', 32: b'@', 33: b'A', 34: b'B', 35: b'C', 36: b'D', 37: b'E', 38: b'F', 39: b'G', 40: b'H', 41: b'I', 42: b'J', 43: b'K', 44: b'L', 45: b'M', 46: b'N', 47: b'O', 48: b'P', 49: b'Q', 50: b'R', 51: b'S', 52: b'T', 53: b'U', 54: b'V', 55: b'W', 56: b'X', 57: b'Y', 58: b'Z', 59: b'[', 60: b'\\', 61: b']', 62: b'^', 63: b'_', 64: b'`', 65: b'a', 66: b'b', 67: b'c', 68: b'd', 69: b'e', 70: b'f', 71: b'g', 72: b'h', 73: b'i', 74: b'j', 75: b'k', 76: b'l', 77: b'm', 78: b'n', 79: b'o', 80: b'p', 81: b'q', 82: b'r', 83: b's', 84: b't', 85: b'u', 86: b'v', 87: b'w', 88: b'x', 89: b'y', 90: b'z', 91: b'{', 92: b'|', 93: b'}', 94: b'~', 95: b'\xa1', 96: b'\xa2', 97: b'\xa3', 98: b'\xa4', 99: b'\xa5', 100: b'\xa6', 101: b'\xa7', 102: b'\xa8', 103: b'\xa9', 104: b'\xaa', 105: b'\xab', 106: b'\xac', 107: b'\xae', 108: b'\xaf', 109: b'\xb0', 110: b'\xb1', 111: b'\xb2', 112: b'\xb3', 113: b'\xb4', 114: b'\xb5', 115: b'\xb6', 116: b'\xb7', 117: b'\xb8', 118: b'\xb9', 119: b'\xba', 120: b'\xbb', 121: b'\xbc', 122: b'\xbd', 123: b'\xbe', 124: b'\xbf', 125: b'\xc0', 126: b'\xc1', 127: b'\xc2', 128: b'\xc3', 129: b'\xc4', 130: b'\xc5', 131: b'\xc6', 132: b'\xc7', 133: b'\xc8', 134: b'\xc9', 135: b'\xca', 136: b'\xcb', 137: b'\xcc', 138: b'\xcd', 139: b'\xce', 140: b'\xcf', 141: b'\xd0', 142: b'\xd1', 143: b'\xd2', 144: b'\xd3', 145: b'\xd4', 146: b'\xd5', 147: b'\xd6', 148: b'\xd7', 149: b'\xd8', 150: b'\xd9', 151: b'\xda', 152: b'\xdb', 153: b'\xdc', 154: b'\xdd', 155: b'\xde', 156: b'\xdf', 157: b'\xe0', 158: b'\xe1', 159: b'\xe2', 160: b'\xe3', 161: b'\xe4', 162: b'\xe5', 163: b'\xe6', 164: b'\xe7', 165: b'\xe8', 166: b'\xe9', 167: b'\xea', 168: b'\xeb', 169: b'\xec', 170: b'\xed', 171: b'\xee', 172: b'\xef', 173: b'\xf0', 174: b'\xf1', 175: b'\xf2', 176: b'\xf3', 177: b'\xf4', 178: b'\xf5', 179: b'\xf6', 180: b'\xf7', 181: b'\xf8', 182: b'\xf9', 183: b'\xfa', 184: b'\xfb', 185: b'\xfc', 186: b'\xfd', 187: b'\xfe', 188: b'\xff', 189: b'\x00', 190: b'\x01', 191: b'\x02', 192: b'\x03', 193: b'\x04', 194: b'\x05', 195: b'\x06', 196: b'\x07', 197: b'\x08', 198: b'\t', 199: b'\n', 200: b'\x0b', 201: b'\x0c', 202: b'\r', 203: b'\x0e', 204: b'\x0f', 205: b'\x10', 206: b'\x11', 207: b'\x12', 208: b'\x13', 209: b'\x14', 210: b'\x15', 211: b'\x16', 212: b'\x17', 213: b'\x18', 214: b'\x19', 215: b'\x1a', 216: b'\x1b', 217: b'\x1c', 218: b'\x1d', 219: b'\x1e', 220: b'\x1f', 221: b' ', 222: b'\x7f', 223: b'\x80', 224: b'\x81', 225: b'\x82', 226: b'\x83', 227: b'\x84', 228: b'\x85', 229: b'\x86', 230: b'\x87', 231: b'\x88', 232: b'\x89', 233: b'\x8a', 234: b'\x8b', 235: b'\x8c', 236: b'\x8d', 237: b'\x8e', 238: b'\x8f', 239: b'\x90', 240: b'\x91', 241: b'\x92', 242: b'\x93', 243: b'\x94', 244: b'\x95', 245: b'\x96', 246: b'\x97', 247: b'\x98', 248: b'\x99', 249: b'\x9a', 250: b'\x9b', 251: b'\x9c', 252: b'\x9d', 253: b'\x9e', 254: b'\x9f', 255: b'\xa0', 256: b'\xad'}
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    # 将 special tokens 模式放在最前面，优先匹配
    PAT = f"{special_pattern}|{base_PAT}"

    # 异常处理
    if not os.path.exists(input_path):
        raise FileExistsError("Do not exist this file,Please check it agagin")
    if vocab_size <= len(special_tokens):
        raise ValueError("Vocab size must greatter than the number of special token")
    
    str_chunks_list = []
    # 文本分段
    with open(input_path,'rb') as f:
        boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            str_chunk_list:list[str] = re.findall(PAT, chunk)
            str_chunks_list.extend(str_chunk_list)

    vocab:dict = {}
    next_id = 0
    merger:tuple[list(bytes,bytes)] = []
    int_words_list:list[list[int]] = []

    for special_token in special_tokens:
        vocab[next_id] = special_token.encode('utf-8')
        next_id += 1
    vocab.update(initialize_vocab)
    next_id = 257

    # 反向词汇表
    reverse_vocab = {v:k for k,v in vocab.items()}
    # 最大添加步数
    max_merge = vocab_size - len(vocab)


    for str_word in str_chunks_list:
        if str_word == '<|endoftext|>':
            continue
        byte_word = str_word.encode('utf-8')
        int_word = list(byte_word)
        int_words_list.append(int_word)



    for _ in range(max_merge):
        # 计算最大pair
        pair_dict = defaultdict(int)
        for int_word in int_words_list:
            if len(int_word) <= 1:
                continue
            for i in range(len(int_word)-1):
                pair = (int_word[i],int_word[i+1])
                pair_dict[pair] += 1

        if not pair_dict:
            break
        
        pair_dict = Counter(pair_dict)
        max_freq = max(pair_dict.values())
        same_freq_pair = [k for k,v in pair_dict.items() if v== max_freq]
        
        if len(same_freq_pair) != 1:    # [(258,269),(32,100)]
            # print(same_freq_pair)
            comparable_pairs_int_byte = []
            for left_int,right_int in same_freq_pair:
                if left_int > 256:
                    left_byte = vocab[left_int]                 
                else: 
                    left_byte  = bytes([left_int])

                if right_int > 256:
                    right_byte = vocab[right_int] 
                else:
                    right_byte = bytes([right_int])
                comparable_pairs_int_byte.append(((left_int,right_int),(left_byte,right_byte)))
            def get_sorted_key(item):
                max_byte = item[1]
                return max_byte

            comparable_pairs_int_byte_sorted = sorted(comparable_pairs_int_byte,key = get_sorted_key,reverse=True)
            # print(comparable_pairs_int_byte_sorted)
            res = [item[0] for item in comparable_pairs_int_byte_sorted ][0]
            # print(f"res{res}")
            most_common_pair = res
        else:
            most_common_pair = pair_dict.most_common(1)[0][0]


        if most_common_pair[0] > 256:
            left_byte = vocab[most_common_pair[0]]
        else:
            left_byte = bytes([most_common_pair[0]])
            # left_byte = vocab[most_common_pair[0]]
        if most_common_pair[1] > 256:
            right_byte = vocab[most_common_pair[1]]
        else:
            right_byte = bytes([most_common_pair[1]])


        new_pair = left_byte+right_byte
        merger.append((left_byte,right_byte))
        vocab[next_id] = new_pair
        reverse_vocab[new_pair] = next_id


        # 合并：
        new_int_words_list = []
        for int_word in int_words_list:
            if len(int_word) <= 1:
                continue
            i = 0
            new_word = []
            while i < len(int_word):
                if i < len(int_word) -1 and int_word[i] == most_common_pair[0] and int_word[i+1] == most_common_pair[1]:
                    new_word.append(next_id)
                    i += 2
                else:
                    new_word.append(int_word[i])
                    i += 1
            new_int_words_list.append(new_word)
        
        int_words_list = new_int_words_list
        next_id += 1


    return vocab,merger
