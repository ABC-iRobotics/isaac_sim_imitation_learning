import os
from glob import glob

from setuptools import find_packages, setup

package_name = "imitation_learning"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("lib", package_name), glob(os.path.join(package_name, "IL_Gym_Env.py"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*"))),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ]
    + glob(os.path.join("share", package_name, "config", "**", "*"), recursive=True),
    install_requires=["setuptools", "gymnasium"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="andrasmakany@gmail.com",
    description="TODO: Package description",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ACT_Actor = imitation_learning.ACT_Actor:main",
            "PI_Actor = imitation_learning.PI_Actor:main",
            "Instructor = imitation_learning.Instructor:main",
        ],
    },
)
