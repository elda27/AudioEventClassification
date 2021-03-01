import json
from pathlib import Path
from typing import Any
import cloudpickle as pickle
import gzip
from base64 import b85encode, b85decode


class PickleDecoder(json.JSONDecoder):
    """JSON with pickle decoder for papermill execution.
    The string object starting with "pickle:///with-base85-encoding" is parsing.
    """

    def decode(self, obj: Any) -> Any:
        obj = super().decode(obj)
        return self.decode_section(obj)

    def decode_section(self, obj: dict):
        result = {}
        for key, value in obj.items():
            if isinstance(value, str):
                if value.startswith('pickle:///'):
                    decoding_data = b85decode(value[10:].encode())
                    if decoding_data[:2] == b'\x1f\x8b':
                        decoding_data = gzip.decompress(decoding_data)
                    result[key] = pickle.loads(decoding_data)
                else:
                    result[key] = Path(value[9:])
            elif isinstance(value, list):
                result[key] = list(self.decode_section({
                    i: v for i, v in enumerate(value)
                }.values()))
            elif isinstance(value, dict):
                result[key] = self.decode_section(value)
            else:
                result[key] = value
        return result


class PickleEncoder(json.JSONEncoder):
    """JSON with pickle encoder for papermill execution.
    Not default json encodable object will encode pickle protocol.
    """

    def default(self, obj: Any) -> Any:
        try:
            if isinstance(obj, Path):
                obj = 'file:///' + str(obj.as_posix())

            return super().default(obj)
        except TypeError:
            data = pickle.dumps(obj)
            if len(data) > 5000:  # If over 5k bytes, the byte data will be compressed
                return 'pickle:///' + b85encode(gzip.compress(data)).decode('ascii')
            else:
                return 'pickle:///' + b85encode(data).decode('ascii')
