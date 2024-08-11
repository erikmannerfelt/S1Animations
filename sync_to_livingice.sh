#!/usr/bin/env bash
set -e
python ./livingice.py
rsync -rhL --info=progress2 for_livingice/* livingice.public:/var/www/static.livingiceproject.com/surge_animations/
