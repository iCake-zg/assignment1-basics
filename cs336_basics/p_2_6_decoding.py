









from typing import Iterable,Iterator
import re



class Tokenizer():
    '''
        Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes
    text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided
    special tokens (appending them to the vocabulary if they aren’t already there)
    '''

    def __init__(self,vocab,merges,spcial_tokens = None) -> None:

        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = spcial_tokens

        self.re_vocab:dict[bytes:int] = {v:k for k,v in vocab.items()}
        if spcial_tokens is not None:
            self.special_tokens = sorted(self.special_tokens,key=len,reverse=True)
            self.spcial_to_id = {tok:self.re_vocab[tok.encode('utf-8')] for tok in spcial_tokens}
            special_pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            self.special_re = re.compile(f'({special_pattern})')

    

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens = None):
        '''
            method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens.
        '''
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None

        vocab = {} # {id:int,token:byte}
        next_id = 0

        if special_tokens is not None:
            for token in special_tokens:
                vocab[next_id] = token.encode('utf-8')
                next_id += 1
                
        with open(vocab_filepath,'r',encoding='utf-8') as f:     
            for _ ,token in enumerate(f):
                token = token.strip()
                vocab[next_id] = token
                next_id += 1
                
        merge = []
        with open(merges_filepath,'r',encoding='utf-8') as f:     
            for line in f:
                a,b = line.strip().split()
                merge.append((a,b))

        return Tokenizer(vocab,merge,special_tokens)
        

    def encode(self,text:str)-> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        ids:list[int] = []

        if self.special_tokens is None:
            text_bytes = text.encode('utf-8') #{byte:int}
            # 拆分成字节形式
            tokens:list[bytes] = [bytes([b]) for b in text_bytes]
            merge_rank = {merge:i for i,merge in enumerate(self.merges)}
            # 找到最佳合并并且合并
            while True:
                min_rank = float('inf')
                pairs = [(tokens[i],tokens[i+1]) for i in range(len(tokens)-1)]
                best_pair = None

                for pair in pairs:      # (byte,byte)
                    if pair in merge_rank and merge_rank[pair] < min_rank:
                        min_rank = merge_rank[pair]
                        best_pair = pair
                
                if best_pair is None:
                    break
                    
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                tokens = new_tokens
            # 将字节形式转换为整数形式
            for token in tokens:
                ids.append(self.re_vocab[token])
        else:
            segments = self.special_re.split(text)
            for seg in segments:
                if seg in self.spcial_to_id:
                    ids.append(self.spcial_to_id[seg])
                else:
                    tokens:list[bytes] = [bytes([b]) for b in seg.encode('utf-8')]
                    merge_rank = {merge:i for i,merge in enumerate(self.merges)}

                    # 找到最佳合并并且合并
                    while True:
                        min_rank = float('inf')
                        pairs = [(tokens[i],tokens[i+1]) for i in range(len(tokens)-1)]
                        best_pair = None

                        for pair in pairs:  
                            if pair == (b'\n',b'\n'):
                                continue    # (byte,byte)
                            if pair in merge_rank and merge_rank[pair] < min_rank:
                                min_rank = merge_rank[pair]
                                best_pair = pair
                        
                        if best_pair is None:
                            break
                            
                        new_tokens = []
                        i = 0
                        while i < len(tokens):
                            if i < len(tokens)-1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                                new_tokens.append(best_pair[0] + best_pair[1])
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1

                        tokens = new_tokens
                    
                    for token in tokens:
                        ids.append(self.re_vocab[token])


        return ids
        



    
    def encode_iterable(self,iterable: Iterable[str]) -> Iterator[int]:
        '''
            strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into
        memory
        '''
        for text in iterable:
            yield from self.encode(text)



    
    def decode(self,ids:list[int])-> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        # 将整数形式转换为字节形式

        byte_string = b"".join(self.vocab[i] for i in ids)
        return byte_string.decode('utf-8')





        
        