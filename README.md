## swarm_frame to Webots2023b

```python
mkdir -p swarm_frame/src

cd swarm_frame/src

git clone --branch swarm_frame https://github.com/SICC-Group/epuck-swarm-webots2023b.git

colcon build --symlink-install

cd ~/swarm_frame/src/swarm_frame/swarm_frame/demo
```

Then, start a Webots simulation, it can contain one or more robots，launch the demo.

`python3 single_robot.py`

 