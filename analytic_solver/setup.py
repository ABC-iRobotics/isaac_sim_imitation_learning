from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'analytic_solver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, package_name, 'moveit_interface'), glob('*.[pxy][yma]*')),
    ] +
    glob(os.path.join('share', package_name, 'config', '**', '*'), recursive=True) #+ 
    ,
    install_requires=['setuptools', 'pillow', 'opencv-contrib-python', 'cv-bridge'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='andrasmakany@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'AnalyticSolver = analytic_solver.AnalyticSolver:main',
            'TrajectoryRecorder = analytic_solver.TrajectoryRecorder:main'
        ],
    },
)
