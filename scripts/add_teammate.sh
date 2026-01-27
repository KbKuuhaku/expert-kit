#!/bin/bash
# Usage: ./add_teammate.sh [username] [user_id]

set -eu 

USERNAME=$1
USER_UID=$2

SSH_BASE_DIR="/home/${USERNAME}/.ssh"

if [ -z "${USERNAME}" ] || [ -z "${USER_UID}" ]; then 
    echo "Username and user id cannot be empty"
    echo "Usage: add_teammate [username] [user_id] [ssh_auth_str]"
    exit 1
fi

## Add user and sudo
if ! id "${USERNAME}" > /dev/null 2>&1; then
    echo "User not found, adding user ${USERNAME} (UID = ${USER_UID})..."
    useradd -m -s /bin/bash -u ${USER_UID} ${USERNAME} # add user and the home directory
    usermod -aG sudo ${USERNAME} # add sudo to user
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    echo "- User ${USERNAME} added!"
else
    echo "- User ${USERNAME} exists, update the SSH authorized keys next"
fi

## SSH settings
# Register SSH public key
echo "Enter your id_rsa.pub:"
read SSH_AUTH_STR
echo "Set up SSH key for ${USERNAME}..."
mkdir -p ${SSH_BASE_DIR}
echo ${SSH_AUTH_STR} >> ${SSH_BASE_DIR}/authorized_keys

# Change permissions
chmod 700 ${SSH_BASE_DIR}
chmod 600 ${SSH_BASE_DIR}/authorized_keys
chown -R ${USERNAME}:${USERNAME} ${SSH_BASE_DIR}   # gives username access to ssh base dir

echo "- SSH authorized keys updated!"
