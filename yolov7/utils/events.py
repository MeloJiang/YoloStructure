#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil


def load_yaml(file_path):
    """从YAML文件中加载数据字典"""
    print(f"Configs loading from yaml file: {file_path}...")
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    print("✅Configs loading completed.\n")
    return data_dict

