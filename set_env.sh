#!/bin/bash

ZSHRC=$HOME/.zshrc
echo "Attempting to append environment variables to $ZSHRC..."

if [[ ! -v LIBTORCH ]]; then
    echo "LIBTORCH not found, add it to zshrc..."
    echo "export LIBTORCH=$(realpath ./vendor/libtorch)" >> $ZSHRC
fi

if [[ ! -v DYLD_FALLBACK_LIBRARY_PATH ]]; then
    echo "DYLD_FALLBACK_LIBRARY_PATH not found, add it to zshrc..."
    echo "export DYLD_FALLBACK_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)" >> $ZSHRC
fi

if [[ ! -v LD_LIBRARY_PATH ]]; then
    echo "LD_LIBRARY_PATH not found, add it to zshrc..."
    echo "export LD_LIBRARY_PATH=$(realpath ./vendor/libtorch/lib)"  >> $ZSHRC
fi

if [[ ! -v EK_CONFIG ]]; then
    echo "EK_CONFIG not found, add it to zshrc..."
    echo "export EK_CONFIG=$(realpath ./dev/hello-world.config.yaml)" >> $ZSHRC
fi

# Restart zsh
exec zsh
