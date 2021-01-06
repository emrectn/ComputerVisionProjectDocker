#!/bin/bash
# -----------------------------------------------------------------------------
# COVID APP INSTALLER
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Create or Update Vision Folder
# -----------------------------------------------------------------------------

if test -d "vision"; then
    echo "vision exists."
	git -C vision/ pull

else
    echo "vision not exists."
	git clone #githublink vision

fi


# -----------------------------------------------------------------------------
# Downloand Weights
# -----------------------------------------------------------------------------

if [ ! -f "vision/models/model.fold_2.best_loss.hdf5" ]; then
    echo "model.fold_2.best_loss.hdf5 Downloading"

    fileid="1ykKaHG4Gk1I3prtzqGxL70tlkPoNEn-F"
    filename="model.fold_2.best_loss.hdf5"

    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

	mv model.fold_2.best_loss.hdf5  vision/models/
else
	echo "weights is exists"
fi

echo "Installation Done"


