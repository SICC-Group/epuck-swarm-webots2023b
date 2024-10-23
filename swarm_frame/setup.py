from setuptools import setup
from glob import glob

package_name = 'swarm_frame'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', [
    'resource/' + package_name
]))
data_files.append(('share/' + package_name, [
    'package.xml'
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
data_files.append(('share/' + package_name + '/config', glob('config/*.py')))


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
            'single_robot = swarm_frame.demo.single_robot:main',
            'single_public_behavior = swarm_frame.demo.single_public_behavior:main',
            'single_private_behavior = swarm_frame.demo.single_private_behavior:main',
            'multi_robots = swarm_frame.demo.multi_robots:main',
            'multi_robots_behavior = swarm_frame.demo.multi_robots_behavior:main',
            'multi_share_behavior = swarm_frame.demo.multi_share_behavior:main',

        ],
    },
)
