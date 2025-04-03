from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'isaac_sim_scene_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ] +
        glob(os.path.join('share', package_name, 'config', '**', '*'), recursive=True),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='andrasmakany@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ScenePlanner = isaac_sim_scene_handler.ScenePlanner:main'
        ],
    },
)
