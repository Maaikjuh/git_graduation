# -*- coding: utf-8 -*-
"""
This module provides classes for data storage and manipulation.
"""
import pandas as pd

__all__ = ['Dataset']


class Dataset(object):
    """
    Dataset objects provide a shared, common, interface for plotting, in-memory
    storage, and file input/output routines.

    Args:
        filename (optional): Path to an existing data (CSV) file to load in.
    """
    def __init__(self, keys=None, filename=None):
        # If a filename was specified then load it.
        if filename:
            self._dataframe = pd.read_csv(filename)
            self._keys = self._dataframe.columns.values.tolist()

        # Otherwise, create a set of storage arrays, one for each key:
        elif keys:
            self._keys = list(keys)
            self._datadict = {k: [] for k in self._keys}

    def __str__(self): # called when str(self) is used
        try:
            return str(self._dataframe)
        except AttributeError:
            return str(pd.DataFrame.from_dict(self._datadict))

    def __getitem__(self, key): # called when self[key] is used
        try:
            return self._dataframe[key]
        except AttributeError:
            return pd.DataFrame.from_dict(self._datadict)[key]

    def append(self, **kwargs):
        """
        Safely append a row of data to the dataset.

        Each data value should be supplied as a key/value pair. Keys that are
        not supplied are given a value of **-1**.
        """
        for k in self.keys():
            self._datadict[k].append(kwargs.get(k, -1))

    def keys(self):
        """
        Return a list of named keys (column names) for this dataset.
        """
        return self._keys

    def save(self, filename):
        """
        Save this dataset to a CSV file.

        Args:
            filename: Name (or path) to save the dataset as/to.
        """
        kwds = {'index': False, 'columns': self.keys()}

        try:
            try:
                self._dataframe.to_csv(filename, **kwds)
            except AttributeError:
                pd.DataFrame.from_dict(self._datadict).to_csv(filename, **kwds)

        except FileNotFoundError:
            import os
            os.mkdir('/'.join(filename.split('/')[:-1]))
            self.save(filename)

    def to_dataframe(self):
        """
        Return this dataset as a :class:`pandas.DataFrame` object.
        """
        try:
            return pd.DataFrame.from_dict(self._datadict)
        except AttributeError:
            return self._dataframe


# class _Dataset(object):
#     def __init__(self, *args, **kwargs):
#         if len(kwargs) == 0:
#             self._keys = self.default_keys() + list(args)
#             self._data = {k: [] for k in self._keys}
#
#         else:
#             keys = self.default_keys()
#             [keys.append(k) for k in kwargs.keys() if k not in keys]
#             self._keys = keys
#             self._data = {k: list(kwargs[k]) for k in self.keys()}
#
#     def items(self):
#         return self._data
#
#     def load(self, filename, delimiter=',', t0=0, t1=None):
#         keys = self.keys()
#         data = np.genfromtxt(filename, delimiter=delimiter)
#         idx0 = int(t0)
#         idx1 = np.abs(data[:, 0] - t1).argmin() + 1 if t1 is not None else None
#         self._data = {k: list(data[idx0:idx1, i]) for i, k in enumerate(keys)}
#
#     def __getitem__(self, item):
#         if isinstance(item, slice) or isinstance(item, int):
#             temp = {k: self._data[k][item] for k in self.keys()}
#             return temp
#
#         elif isinstance(item, tuple) and isinstance(item[0], np.ndarray):
#             # passed output from np.where()
#             temp = {k: self._data[k][item[0]] for k in self.keys()}
#             return temp
#
#         else:
#             return self._data[item]
#
#     def __setitem__(self, key, value):
#         if isinstance(key, slice) and key.start is key.stop is key.step:
#             # passed [:] for slice
#             self._data = {k: list(value[k]) for k in self.keys()}
#
#         else:
#             self._keys.append(key)
#             self._data[key] = value
#
#     def __repr__(self):
#         data = ['{k}={v}'.format(k=k, v=v) for k, v in self._data.items()]
#         return self.__class__.__name__ + '(' + ', '.join(data) + ')'
