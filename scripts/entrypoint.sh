#!/bin/bash

# Move into "/etc/environment" all environment variables defined in dockerfile
# /etc/environment is like a global KEY-VALUE pairs which could be defined in .bashrc
echo "Syncing ENV from the Dockerfile..."
env | grep -v "^HOME" | grep -v "^PWD" >> "/etc/environment"

# Start SSHD
echo "Starting SSHD..."
exec /usr/sbin/sshd -D
