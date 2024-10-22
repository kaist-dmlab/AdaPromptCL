# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# https://github.com/pytorch/vision/blob/8635be94d1216f10fb8302da89233bd86445e449/torchvision/datasets/utils.py

import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile
import numpy as np
import torch
import codecs

from torch.utils.model_zoo import tqdm


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def iterable_to_str(iterable):
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def get_cls2supcls_cifar100():
    return {0: 4,
            1: 1,
            2: 14,
            3: 8,
            4: 0,
            5: 6,
            6: 7,
            7: 7,
            8: 18,
            9: 3,
            10: 3,
            11: 14,
            12: 9,
            13: 18,
            14: 7,
            15: 11,
            16: 3,
            17: 9,
            18: 7,
            19: 11,
            20: 6,
            21: 11,
            22: 5,
            23: 10,
            24: 7,
            25: 6,
            26: 13,
            27: 15,
            28: 3,
            29: 15,
            30: 0,
            31: 11,
            32: 1,
            33: 10,
            34: 12,
            35: 14,
            36: 16,
            37: 9,
            38: 11,
            39: 5,
            40: 5,
            41: 19,
            42: 8,
            43: 8,
            44: 15,
            45: 13,
            46: 14,
            47: 17,
            48: 18,
            49: 10,
            50: 16,
            51: 4,
            52: 17,
            53: 4,
            54: 2,
            55: 0,
            56: 17,
            57: 4,
            58: 18,
            59: 17,
            60: 10,
            61: 3,
            62: 2,
            63: 12,
            64: 12,
            65: 16,
            66: 12,
            67: 1,
            68: 9,
            69: 19,
            70: 2,
            71: 10,
            72: 0,
            73: 1,
            74: 16,
            75: 12,
            76: 9,
            77: 13,
            78: 15,
            79: 13,
            80: 16,
            81: 19,
            82: 2,
            83: 4,
            84: 6,
            85: 19,
            86: 5,
            87: 5,
            88: 8,
            89: 19,
            90: 18,
            91: 1,
            92: 2,
            93: 15,
            94: 6,
            95: 0,
            96: 17,
            97: 8,
            98: 14,
            99: 13}
    
def get_cls2supcls_imr():
    return {0: 0,1: 0,2: 0,3: 0,4: 1,5: 1,6: 1,7: 1,8: 1,9: 1,10: 2,11: 2,12: 2,13: 2,14: 2,15: 2,
            16: 2,17: 5,18: 5,19: 1,20: 1,21: 1,22: 1,23: 1,24: 1,25: 1,26: 4,27: 0,28: 5,29: 5,30: 5,
            31: 1,32: 1,33: 1,34: 1,35: 0,36: 0,37: 0,38: 3,39: 3,40: 3,41: 3,42: 3,43: 3,44: 3,45: 3,
            46: 3,47: 3,48: 3,49: 3,50: 3,51: 3,52: 3,53: 3,54: 3,55: 3,56: 3,57: 3,58: 3,59: 3,60: 3,
            61: 3,62: 3,63: 3,64: 3,65: 3,66: 3,67: 3,68: 3,69: 4,70: 4,71: 4,72: 4,73: 4,74: 4,75: 4,
            76: 4,77: 4,78: 4,79: 4,80: 5,81: 5,82: 5,83: 5,84: 5,85: 5,86: 5,87: 5,88: 5,89: 0,90: 4,
            91: 4,92: 4,93: 4,94: 4,95: 4,96: 4,97: 4,98: 4,99: 4,100: 4,101: 4,102: 4,103: 4,104: 4,
            105: 4,106: 4,107: 4,108: 4,109: 0,110: 0,111: 0,112: 6,113: 7,114: 9,115: 8,116: 11,117: 9,
            118: 12,119: 9,120: 11,121: 9,122: 9,123: 11,124: 8,125: 9,126: 9,127: 9,128: 9,129: 9,
            130: 7,131: 11,132: 11,133: 9,134: 8,135: 6,136: 7,137: 6,138: 9,139: 6,140: 9,141: 9,
            142: 6,143: 6,144: 9,145: 7,146: 9,147: 8,148: 9,149: 9,150: 11,151: 9,152: 9,153: 9,
            154: 7,155: 7,156: 9,157: 12,158: 8,159: 6,160: 7,161: 9,162: 9,163: 12,164: 7,165: 5,
            166: 7,167: 8,168: 7,169: 9,170: 12,171: 9,172: 6,173: 9,174: 6,175: 7,176: 9,177: 10,
            178: 10,179: 10,180: 10,181: 10,182: 10,183: 10,184: 10,185: 10,186: 10,187: 10,188: 10,
            189: 10,190: 10,191: 10,192: 10,193: 10,194: 10,195: 10,196: 13,197: 12,198: 0,199: 10}