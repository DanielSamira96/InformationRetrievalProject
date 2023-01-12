from collections import Counter
import itertools
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing

BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket. 
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def write(self, b, base_dir):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self.upload_to_gcp(base_dir)
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((base_dir + "/" + self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def upload_to_gcp(self, base_dir):
        file_name = self._f.name
        blob = self.bucket.blob(f"{base_dir}/{file_name}")
        blob.upload_from_filename(file_name)


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1


class InvertedIndex:
    def __init__(self, docs={}):
        self.df = Counter()
        self.DL = Counter()
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)

    def write_index(self, base_dir, name):
        # GLOBAL DICTIONARIES
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self, query):
        with closing(MultiFileReader()) as reader:
            for term in query:
                if term in self.posting_locs:
                    locs = self.posting_locs[term]
                    b = reader.read(locs, self.df[term] * TUPLE_SIZE)
                    posting_list = []
                    for i in range(self.df[term]):
                        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                        posting_list.append((doc_id, tf))
                    yield term, posting_list

    def posting_lists_iter_anchor(self, query):
        with closing(MultiFileReader()) as reader:
            for term in query:
                if term in self.posting_locs:
                    locs = self.posting_locs[term]
                    b = reader.read(locs, self.df[term] * 4)
                    posting_list = []
                    for i in range(self.df[term]):
                        doc_id = int.from_bytes(b[i * 4:i * 4 + 4], 'big')
                        posting_list.append(doc_id)
                    yield term, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(".", bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b, base_dir)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp(base_dir)
            InvertedIndex._upload_posting_locs(bucket_id, base_dir, posting_locs, bucket_name)
        return bucket_id

    @staticmethod
    def write_a_posting_list_anchor(b_w_pl, base_dir, bucket_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(".", bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([doc_id.to_bytes(4, 'big') for doc_id in pl])
                # write to file(s)
                locs = writer.write(b, base_dir)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp(base_dir)
            InvertedIndex._upload_posting_locs(bucket_id, base_dir, posting_locs, bucket_name)
        return bucket_id

    @staticmethod
    def _upload_posting_locs(bucket_id, base_dir, posting_locs, bucket_name):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"{base_dir}/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")
