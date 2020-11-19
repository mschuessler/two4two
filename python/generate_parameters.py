import os, sys

from parameters import Parameters

if __name__ == '__main__':
    parameters = Parameters()
    # n
    # save_location
    # object_types
    # structure_types
    parameters.generate_many(sys.argv[-4],
                             sys.argv[-3],
                             sys.argv[-2],
                             sys.argv[-1])