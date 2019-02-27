# -*- coding: utf-8 -*-
from setuptools import setup

setup_requires = [
]

install_requires = [
    'tensorflow >= 1.12.0',
]

dependency_links = [
]

setup(
    name='furiosa-ssd-tf',
    version='0.1',
    description='Furiosa ML Framework, setuptools version of https://github.com/balancap/SSD-Tensorflow',
    author='Sol Kim',
    author_email='skim@furiosa.ai',
    test_suite='tests',
    packages=[
        'furiosa_ssd_tf',
        'furiosa_ssd_tf.nets',
        'furiosa_ssd_tf.preprocessing',
        'furiosa_ssd_tf.tf_extended',
        'furiosa_ssd_tf.datasets',
        'furiosa_ssd_tf.deployment',
        'nets',
        'preprocessing',
        'tf_extended',
        'datasets',
        'deployment',
    ],
    package_dir={
        'furiosa_ssd_tf': 'furiosa_ssd_tf',
        'nets': 'furiosa_ssd_tf/nets',
        'preprocessing': 'furiosa_ssd_tf/preprocessing',
        'tf_extended': 'furiosa_ssd_tf/tf_extended',
        'datasets': 'furiosa_ssd_tf/datasets',
        'deployment': 'furiosa_ssd_tf/deployment',
    },
    install_requires=install_requires,
    tests_require=install_requires,
    setup_requires=setup_requires,
    dependency_links=dependency_links,
    project_urls={
        "Bug Tracker": "https://github.com/furiosa-ai/furiosa-ssd-tf/issues",
        "Documentation": "https://furiosa.ai/",
        "Source Code": "https://github.com/furiosa-ai/furiosa-ssd-tf/",
    },
    exclude_package_data={
        '': [
            '.gitignore',
        ],
    },
)
