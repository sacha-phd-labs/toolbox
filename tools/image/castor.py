import numpy as np
import os

def read_castor_binary_file(file_path, reader='numpy', return_metadata=False):
    """
    file_path: str
        Path to the binary hdr file.
    We assume that this header file contains all necessary metadata in a known castor format.
    """

    if not file_path.endswith('.hdr'):
        file_path += '.hdr'

    metadata = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ' := ' in line:
                key, value = line.split(' := ', 1)
                metadata[key.strip()] = value.strip()

    binary_file_path = metadata.get('!name of data file')
    number_format = metadata.get('number format', None) or metadata.get('!number format', None)
    dim_x = metadata.get('matrix size [1]', None) or metadata.get('!matrix size [1]', None)
    dim_y = metadata.get('matrix size [2]', None) or metadata.get('!matrix size [2]', None)
    dim_z = metadata.get('matrix size [3]', '1') or metadata.get('!matrix size [3]', '1')

    assert binary_file_path is not None, "Binary data file path not found in header."
    assert number_format is not None, "Number format not found in header."
    assert dim_x is not None and dim_y is not None, "Matrix size not found in header."

    with open(os.path.join(os.path.dirname(file_path), binary_file_path), 'rb') as f:
        if reader == 'numpy':
            if 'float' in number_format:
                data = np.frombuffer(f.read(), dtype=np.float32)
            elif 'signed integer' in number_format:
                data = np.frombuffer(f.read(), dtype=np.int16)
            else:
                raise ValueError(f"Unsupported number format: {number_format}")
            data = data.reshape((int(dim_z), int(dim_y), int(dim_x)))
        else:
            raise ValueError(f"Unsupported reader: {reader}")
    
    if return_metadata:
        return data, metadata
    else:
        return data

def write_binary_file(file_path, data: np.ndarray, metadata={}, binary_extension=''):
    """
    file_path: str
        Path to the binary hdr file.
    data: np.ndarray
        Data to write.
    metadata: dict
        Metadata addons to write in the header file.
    """

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    if not file_path.endswith('.hdr'):
        file_path += '.hdr'

    dtype = data.dtype
    shape = data.shape
    #
    metadata.update({
        '!name of data file': os.path.basename(file_path).replace('.hdr', binary_extension),
        '!number format': 'float' if 'float32' == str(dtype) else 'signed integer' if str(dtype) == 'int16' else 'unknown',
        '!matrix size [1]': str(shape[2]) if len(shape) > 2 else str(shape[1]),
        '!matrix size [2]': str(shape[1]) if len(shape) > 2 else str(shape[0]),
        '!matrix size [3]': str(shape[0]) if len(shape) > 2 else '1',
    })

    with open(os.path.join(file_path.replace('.hdr', binary_extension)), 'wb') as f:
        f.write(data.tobytes())

    with open(file_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key} := {value}\n")

