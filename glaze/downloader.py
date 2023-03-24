import random
from pathlib import Path
import os
import zipfile
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve
import shutil
import six
import errno
import hashlib
home_path = Path.home()
projects_root_path = os.path.join(home_path, '.glaze')
if not os.path.isdir(projects_root_path):
    os.mkdir(projects_root_path)
    
def download_all_resources(signal):
    get_file(os.path.join(projects_root_path), 'http://mirror.cs.uchicago.edu/fawkes/files/glaze/base.zip', '0404aa8a44342abb4de336aafa4878e6', '1 / 9', True, signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'extract', 'signal'))
    get_file(os.path.join(projects_root_path, 'base', 'base', 'unet'), 'http://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin', 'f54896820e5730b03996ce8399c3123e', '2 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(os.path.join(projects_root_path, 'base', 'base', 'text_encoder'), 'http://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin', '167df82281473d0f2a320aea8fab9059', '3 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(os.path.join(projects_root_path, 'base', 'base', 'vae'), 'http://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.bin', 'a90d1567d06336ea076afe50455c0712', '4 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(projects_root_path, 'http://mirror.cs.uchicago.edu/fawkes/files/glaze/bpe_simple_vocab_16e6.txt.gz', '933b7abbbbde62c36f02f0e6ccde464f', '5 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(projects_root_path, 'http://mirror.cs.uchicago.edu/fawkes/files/glaze/preview_mask.p', '120c5eb7a6e6928405e58e5fe34886d5', '6 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(projects_root_path, 'http://mirror.cs.uchicago.edu/fawkes/files/glaze/glaze-qc.p', '0f3a00b66b463a908e442e2ba43ce464', '7 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(projects_root_path, 'http://mirror.cs.uchicago.edu/fawkes/files/glaze/glaze.p', '869bd38b0079b4ede3f5fe4f0e19ae22', '8 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))
    get_file(projects_root_path, 'http://mirror.cs.uchicago.edu/fawkes/files/glaze/clip_model.p', '41c6e336016333b6210b9840d1283d9f', '9 / 9', signal, **('root_dir', 'origin', 'md5_hash', 'file_num', 'signal'))


def get_file(origin, root_dir, md5_hash, file_hash, hash_algorithm, extract, archive_format, file_num, signal = (None, None, 'auto', False, 'auto', None, None)):
    class ProgressTracker():
        progbar = None

    def dl_progress(count, block_size, total_size):
        if random.uniform(0, 1) < 0.05:
            signal.emit(
                "download=Downloading resource {}\n({:.2f} / {:.2f} Mb)".format(
                    file_num,
                    count * block_size / 1024 / 1024,
                    total_size / 1024 / 1024,
                )
            )
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    fname = origin.split('/')[-1]
    fpath = os.path.join(root_dir, fname)
    download = False

    if os.path.exists(fpath):

        if file_hash is not None:
            if not validate_file(fpath, file_hash, hash_algorithm):
                download = True
        else:
            download = True
    else:
        download = True

    if download:

        error_msg = 'URL fetch failure on {}: {} -- {}'

        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        finally:
            if os.path.exists(fpath):
                os.remove(fpath)

    if download and extract:
        _extract_archive(fpath)
        

    return fpath


def _extract_archive(file_path, path):
    import zipfile
    from shutil import move

    open_fn = zipfile.ZipFile

    if not file_path.lower().endswith('.zip'):
        raise AssertionError

    tmp = file_path.replace('.zip', '0')

    with open_fn(file_path, 'r') as f:
        f.extractall(tmp)

    outp = file_path.replace('.zip', '')

    move(tmp, outp)

    return None



def _makedirs_exist_ok(datadir):
    if six.PY2:
        try:
            os.makedirs(datadir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
    else:
        os.makedirs(datadir, exist_ok=True)
    
    return None


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """
    Validates a file against a sha256 or md5 hash.
    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        Whether the file is valid
    """
    
    def _hash_file(fpath, algorithm='auto', chunk_size=65535):
        if algorithm == 'sha256' or (algorithm == 'auto' and len(file_hash) == 64):
            import hashlib
            hasher = hashlib.sha256()
        else:
            import hashlib
            hasher = hashlib.md5()

        with open(fpath, 'rb') as fpath_file:
            for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
                hasher.update(chunk)

        return hasher.hexdigest()

    if algorithm == "sha256" or (algorithm == "auto" and len(file_hash) == 64):
        hasher = "sha256"
    else:
        hasher = "md5"

    hashed_fpath = _hash_file(fpath, hasher, chunk_size)

    return len(hashed_fpath) == len(file_hash) and hashed_fpath == file_hash

