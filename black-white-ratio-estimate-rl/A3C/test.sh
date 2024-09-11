#!/bin/bash

# 获取脚本的绝对路径
SCRIPT_PATH=$(realpath "$BASH_SOURCE")

# 获取脚本所在的文件夹
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

echo "Script Path: $SCRIPT_PATH"
echo "Script Directory: $SCRIPT_DIR"