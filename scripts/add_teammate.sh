#!/bin/bash

# Usage: ./add_teammate.sh [username] [user_id] [ssh_auth_str]

USERNAME=$1
USER_UID=$2
SSH_AUTH_STR=$3

SSH_BASE_DIR=/home/${USERNAME}/.ssh

if [ -z "${USERNAME}" ] || [ -z "${USER_UID}" ]; then 
    echo "Username and user id cannot be empty"
    echo "Usage: add_teammate [username] [user_id] [ssh_auth_str]"
    exit 1
fi

## Add user and sudo
echo "Add user $1 (UID = $2)..."
useradd -m -s /bin/bash -u ${USER_UID} ${USERNAME} # add user and the home directory
usermod -aG sudo ${USERNAME} # add sudo to user
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

## SSH settings

# Register SSH public key
if [ -n "${SSH_AUTH_STR}" ]; then
    echo "Set up SSH key for ${USERNAME}..."
    mkdir -p ${SSH_BASE_DIR}
    echo ${SSH_AUTH_STR} >> ${SSH_BASE_DIR}/authorized_keys

    # Change permissions
    chmod 700 ${SSH_BASE_DIR}
    chmod 600 ${SSH_BASE_DIR}/authorized_keys
    chown -R ${USERNAME}:${USERNAME} ${SSH_BASE_DIR}   # gives username access to ssh base dir
fi

echo "Successfully added user ${USERNAME} and set up the ssh authorized keys!"
