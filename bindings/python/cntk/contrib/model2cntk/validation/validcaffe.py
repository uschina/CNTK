# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
# import shutil
import numpy as np

from abc import ABCMeta, abstractmethod

CAFFE_RUNTIME = True
try:
    import caffe
except ImportError:
    CAFFE_RUNTIME = False


class ValidCore(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def execute(source_solver, valid_dir):
        pass


class CaffeValidCore(ValidCore):
    @staticmethod
    def execute(source_solver, valid_dir, val_inputs=dict()):
        if not CAFFE_RUNTIME:
            sys.stdout.write('no caffe runtime support, ignore valid...\n')
            return
        sys.stdout.write('start valid feature map...\n')
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(source_solver.model_path, source_solver.weights_path, caffe.TEST)
        for name in net.inputs:
            input_blob = net.blobs[name]
            if name in val_inputs.keys() and val_inputs[name][1] != []:
                # TODO: Only for PSRoiPooling
                target_array = np.array(val_inputs[name][1]).reshape(input_blob.data.shape)
            else:
                target_array = np.load(os.path.join(valid_dir, name + '.npy')).reshape(input_blob.data.shape)
            np.copyto(input_blob.data, target_array)
        net.forward()
        for file_name in os.listdir(valid_dir):
            target, _ = os.path.splitext(file_name)
            if target in net.inputs:
                continue
            gt_result = np.load(os.path.join(valid_dir, file_name))
            test_result = net.blobs[target].data
            power_error = np.power(gt_result.flatten() - test_result.flatten(), 2).sum()
            rsme_diff = np.sqrt(power_error / gt_result.size)
            sys.stdout.write('valid %s with RSME = %s, MAX = %s, MIN = %s\n' %
                             (target, str(rsme_diff), str(gt_result.max()), str(gt_result.min())))
        sys.stdout.write('valid finished...\n')
        # shutil.rmtree(valid_dir)
