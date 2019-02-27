# -*- coding: utf-8 -*-
from unittest import TestCase

# Import furiosa-ssd-tf libraries
import furiosa_ssd_tf.nets.nets_factory


class TestNetsFactory(TestCase):
    def test_networks_map(self):
        self.assertTrue('ssd_512_mobilenet_v2' in furiosa_ssd_tf.nets.nets_factory.networks_map)
