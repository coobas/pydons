from hdf5storage.Marshallers import TypeMarshaller
# import pydons
import h5py

from hdf5storage.utilities import *
# from hdf5storage import lowlevel
from hdf5storage.lowlevel import write_data, read_data


class MatStructMarshaller(TypeMarshaller):
    def __init__(self, MatStructType):
        TypeMarshaller.__init__(self)
        self.python_attributes |= set(['Python.Fields'])
        self.matlab_attributes |= set(['MATLAB_class'])
        self.types = [MatStructType]
        self.python_type_strings = ['{}.{}'.format(MatStructType.__module__, MatStructType.__name__)]
        self.__MATLAB_classes = {MatStructType: 'struct'}
        # Set matlab_classes to empty since NumpyScalarArrayMarshaller
        # handles Groups by default now.
        self.matlab_classes = list(self.__MATLAB_classes.values())

    def write(self, f, grp, name, data, type_string, options):
        # If the group doesn't exist, it needs to be created. If it
        # already exists but is not a group, it needs to be deleted
        # before being created.

        if name not in grp:
            grp.create_group(name)
        elif not isinstance(grp[name], h5py.Group):
            del grp[name]
            grp.create_group(name)

        grp2 = grp[name]

        # Write the metadata.
        self.write_metadata(f, grp, name, data, type_string, options)

        # Delete any Datasets/Groups not corresponding to a field name
        # in data if that option is set.

        if options.delete_unused_variables:
            for field in set([i for i in grp2]).difference(
                    set([i for i in data])):
                del grp2[field]

        # Go through all the elements of data and write them. The H5PATH
        # needs to be set as the path of grp2 on all of them if we are
        # doing MATLAB compatibility (otherwise, the attribute needs to
        # be deleted).
        for k, v in data.items():
            k = unicode(k)
            write_data(f, grp2, k, v, None, options)
            if k in grp2:
                if options.matlab_compatible:
                    set_attribute_string(grp2[k], 'H5PATH', grp2.name)
                else:
                    del_attribute(grp2[k], 'H5PATH')

    def write_metadata(self, f, grp, name, data, type_string, options):
        # First, call the inherited version to do most of the work.

        TypeMarshaller.write_metadata(self, f, grp, name, data,
                                      type_string, options)

        # If we are storing python metadata, we need to set the
        # 'Python.Fields' Attribute to be all the keys. They will be
        if options.store_python_metadata:
            fields = list(data.keys())
            set_attribute_string_array(grp[name], 'Python.Fields',
                                       fields)

        # If we are making it MATLAB compatible, the MATLAB_class
        # attribute needs to be set for the data type. If the type
        # cannot be found or if we are not doing MATLAB compatibility,
        # the attributes need to be deleted.

        tp = type(data)
        if options.matlab_compatible and tp in self.types \
                and tp in self.__MATLAB_classes:
            set_attribute_string(grp[name], 'MATLAB_class',
                                 self.__MATLAB_classes[tp])
            # TODO this does not work because Matlab stores field names
            # in a weird way (H5T_VLEN(:,:) array of H5T_STRING(1))
            # set_attribute_string_array(grp[name], 'MATLAB_fields',
            #                            fields)
        else:
            del_attribute(grp[name], 'MATLAB_class')

        # Write an array of all the fields to the attribute that lists
        # them.
        #
        # NOTE: Can't make it do a variable length set of strings like
        # MATLAB likes. However, not including them seems to cause no
        # problem.
        #
        # set_attribute_string_array(grp[name], \
        #     'MATLAB_fields', [k for k in data])

    def read(self, f, grp, name, options):
        # If name is not present or is not a Group, then we can't read
        # it and have to throw an error.
        if name not in grp or not isinstance(grp[name], h5py.Group):
            raise NotImplementedError('No Group ' + name +
                                      ' is present.')

        # Starting with an empty dict, all that has to be done is
        # iterate through all the Datasets and Groups in grp[name] and
        # add them to the dict with their name as the key. Since we
        # don't want an exception thrown by reading an element to stop
        # the whole reading process, the reading is wrapped in a try
        # block that just catches exceptions and then does nothing about
        # them (nothing needs to be done).
        data = self.types[0]()
        for k in grp[name]:
            # We must exclude group_for_references
            if grp[name][k].name == options.group_for_references:
                continue
            try:
                data[k] = read_data(f, grp[name], k, options)
            except:
                pass
        return data
