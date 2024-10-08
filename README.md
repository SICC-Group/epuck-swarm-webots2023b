
# swarm-epucks
This repository contains typical environments and demo codes for epucks swarm simulation.

## Requirements
Ubuntu 20.04, Webots 2023b, python 3.8

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

## Acknowledgements

Part of this repository is original from the hashgraph-based framework. The corresponding paper is [here](https://link.springer.com/article/10.1007/s10846-022-01759-1). The source code is [here](https://github.com/LuoJie996/swarm-hashgraph). Demo Video is [here](https://pan.baidu.com/s/1mtrY0Fwr6RJt6HVxyEfKdA?pwd=s6s5#list/path=%2F).
