import h5py
import tensorflow as tf
import numpy as np

class RandomBatchSamplerTF:
    def __init__(self, dataset_size, batch_size, shuffle=False, drop_last=False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.n_batches = self.dataset_size // self.batch_size
        self.nonzero_last_batch = (self.dataset_size / self.batch_size) > self.n_batches

    def get_batch_indices(self):
        # Create batch indices (list of slice objects)
        full_batches = [
            slice(i * self.batch_size, (i + 1) * self.batch_size)
            for i in range(self.n_batches)
        ]

        if not self.drop_last and self.nonzero_last_batch:
            full_batches.append(slice(self.n_batches * self.batch_size, self.dataset_size))

        if self.shuffle:
            np.random.shuffle(full_batches)

        return full_batches

    def get_tf_dataset(self, h5_data):
        """Assumes h5_data is an array-like object supporting slicing (like h5py dataset)."""
        batch_slices = self.get_batch_indices()

        def generator():
            for s in batch_slices:
                yield h5_data[s]

        output_shape = (self.batch_size,) + h5_data.shape[1:]
        output_dtype = tf.as_dtype(h5_data.dtype)

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=(None,) + h5_data.shape[1:], dtype=output_dtype)
        )


class HDF5DatasetTF:
    def __init__(self, file_path, key="Particles"):
        self.file_path = file_path
        self.file = h5py.File(self.file_path, 'r')
        self.dss = self.file[key]

        # Allocate empty array for temporary reads
        self.empty_array = np.empty(self.dss.shape, dtype=self.dss.dtype)

    def __len__(self):
        return self.dss.shape[0]  # Number of particles

    def __getitem__(self, object_idx):
        """Supports numpy-style slicing (e.g. dataset[10:20])"""
        if isinstance(object_idx, int):
            object_idx = slice(object_idx, object_idx + 1)

        # Resize empty array for this batch
        batch = self.empty_array
        shape = (object_idx.stop - object_idx.start,) + self.dss.shape[1:]
        batch.resize(shape, refcheck=False)

        # Read from HDF5 dataset
        self.dss.read_direct(batch, source_sel=object_idx)

        return batch

    def get_all_batch_slices(self, batch_size, shuffle=True, drop_last=False):
        sampler = RandomBatchSamplerTF(
            dataset_size=len(self),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return sampler.get_batch_indices()

    def get_tf_dataset(self, batch_size=64, shuffle=True, drop_last=False, batch_slices=None):
        if batch_slices is None:
            batch_slices = self.get_all_batch_slices(batch_size, shuffle, drop_last)

        def generator():
            for s in batch_slices:
                yield self[s]

        sample_shape = self[0:1].shape[1:]

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=(None,) + sample_shape, dtype=tf.float64)
        )

    
    def close(self):
        self.file.close()    


def load_data(file_path, batch_size=64, shuffle=True, drop_last=False):
    dataset = HDF5DatasetTF(file_path)

    batch_slices = dataset.get_all_batch_slices(batch_size=batch_size,
                                                    shuffle=False,
                                                    drop_last=drop_last)
    
    tf_dataset = dataset.get_tf_dataset(batch_size=batch_size,
                                        shuffle=shuffle,
                                        drop_last=drop_last)

    return tf_dataset.prefetch(tf.data.AUTOTUNE), len(batch_slices)
