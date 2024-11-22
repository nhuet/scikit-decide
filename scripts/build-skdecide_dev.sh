#!/bin/bash
set -ex

yum install --setopt=sslverify=false -y git zlib-devel
