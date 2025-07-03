from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'openfusion_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['openfusion_ros', 'openfusion_ros.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yml')),
    ],
    package_data={
        'openfusion_ros': [
            'zoo/xdecoder_seem/configs/seem/*.yaml',
            'zoo/xdecoder_seem/checkpoints/*.pt',
        ]
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='kevin-eppacher@hotmail.de',
    description='ROS 2 integration of OpenFusion for real-time semantic SLAM using SEEM and visual-language queries.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'openfusion_ros = openfusion_ros.openfusion_ros_node:main',
        ],
    },
)
