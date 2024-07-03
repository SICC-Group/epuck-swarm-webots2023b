from setuptools import setup
from glob import glob

package_name = 'webots'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', [
    'resource/' + package_name
]))
data_files.append(('share/' + package_name, [
    'package.xml'
]))
data_files.append(('share/' + package_name, [
    'launch/random_pattern_launch.py',
    'launch/obstacle_avoidance_launch.py',
    'launch/attraction_pattern_launch.py',
    'launch/dispersion_pattern_launch.py',
    'launch/test_launch.py',
    'launch/formation_pattern_launch.py',
    'launch/behavioral_flow_pattern_launch.py',
]))
data_files.append(('share/' + package_name + '/worlds', [
    'worlds/epuck_world.wbt',
    'worlds/mul_epuck_world.wbt',
    'worlds/leader_followers.wbt',
]))
data_files.append(('share/' + package_name + '/protos', [
    'protos/E-puck_enu.proto',
    'protos/E-puckDistanceSensor_enu.proto',
    'protos/LegoTallInterval.proto',
    'protos/LegoTallWall.proto'
]))
data_files.append(('share/' + package_name + '/protos/icons', glob('protos/icons/*')))
data_files.append(('share/' + package_name + '/protos/textures', glob('protos/textures/*')))

data_files.append(('share/' + package_name + '/config', glob('config/*')))
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jzx',
    maintainer_email='jzx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'random_pattern  =  webots.random_pattern:main',
            'obstacle_avoidance  =  webots.obstacle_avoidance:main',
            'attraction_pattern  =  webots.attraction_pattern:main',
            'dispersion_pattern  =  webots.dispersion_pattern:main',
            'formation_pattern = webots.formation_pattern:main',
            'behavioral_flow_pattern = webots.behavioral_flow_pattern:main',
            'webots_differential_drive_node  =  webots.webots_differential_drive_node:main',
            'test  =  webots.test:main',
            'test2  =  webots.test2:main',
            'test3  =  webots.test3:main',
        ],
    },
)
