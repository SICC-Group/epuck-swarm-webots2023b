**migrate swarm-hashgraph to webots2023b**

# swarm-hashgraph
This repository is an implementation of the framework for hashgraph-based swarm robotics.
There are two case studies in this repository, one is black-white ratio estimation and the other is object searching.

The corresponding article: A Fast and Robust Solution for Common Knowledge Formation in Decentralized Swarm Robots(https://link.springer.com/article/10.1007/s10846-022-01759-1)

The source code address of this repository: https://github.com/LuoJie996/swarm-hashgraph
## Requirements
Ubuntu18.04, Webots 2023b, python 3.8

Requirements of py-swirld installation refers to https://github.com/Lapin0t/py-swirld
```
pip install py-swirld
```
## Install
###
    git clone git@github.com:SICC-Group/swarm-hashgraph-webots2023b.git
## Usage
object-searching:
1. Launch the Hashgrpah process
###
    cd object_searching/controllers/py-swirld-object-searching/
    python swirld_object_searching2.py    
2. Launch the Webots simulator by click the webots icon
3. In Webots simulator, open the world object_searching2.wbt in object_searching/worlds/
4. Click the run button

black-white-ratio-estimate:
1. Launch the Hashgrpah process
###
    cd black-white-ratio-estimate/controllers/py-swirld-black-ratio/
    python swirld_black_white_ratio_estimate.py    
2. Launch the Webots simulator by click the webots icon
3. In Webots simulator, open the world epuck_gezi_twenty_48.wbt in black-white-ratio-estimate/worlds/
4. Click the run button
## Tips
Don't forget to change the absolute path of the document in the code to read your preset parameters and save your experimental results:

Replace all paths in the form of '/home/<>/<>.txt' with the paths on your own computer.

All txt files with names that appear in the code can be created as empty files or have 0 filled in on the first line.
## Demo Video
https://pan.baidu.com/s/1mtrY0Fwr6RJt6HVxyEfKdA?pwd=s6s5#list/path=%2F

## Swarm_behavior use
git clone swarm_behavior and interfaces package

then, apt install ros2_webots package
