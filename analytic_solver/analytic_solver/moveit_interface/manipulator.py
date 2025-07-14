from typing import List
import os
import yaml
import subprocess

result = subprocess.check_output("ros2 pkg prefix data_generator",shell = True, text = True)
result = result.split("/install",1)[0]

with open(os.path.join(result,'src/config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

MOVE_GROUP_ARM: str = str(config['moveit_configs']['move_group_arm'])

def joint_names() -> List[str]:
    my_list = []
    for i in config['moveit_configs']['joint_names']:
        my_list.append(str(i))
    return my_list

def base_link_name() -> str:
    return str(config['moveit_configs']['base_link_name'])


def end_effector_name() -> str:
    return str(config['moveit_configs']['end_effector_name'])
