echo "===== Requirements ====="


# sudo update

# sudo apt update
NCCL_PATH=$(find /usr/lib /usr/lib/x86_64-linux-gnu -name "libnccl.so" 2>/dev/null | head -n 1)
if [ NCCL_PATH == "" ]; then
    echo "NCCL library not found."
    if [ "$(nvcc --version | sed -n 's/.*release \([^,]*\),.*/\1/p') | cut -d. -f1" -lt 12 ]; then
        echo "CUDA VERSION NOT SUFFICIENT"
    
    else
        echo "Download NCCL: [y/n]" 
        read nccl_download_allow
        if [ "$nccl_download_allow" == "y" ]; then
            # sudo apt update
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
            sudo dpkg -i cuda-keyring_1.1-1_all.deb
            sudo apt-get update
            sudo apt install libnccl2=2.29.7-1+cuda12.9 libnccl-dev=2.29.7-1+cuda12.9
            echo "NCCL_PATH after download: $(find /usr/lib /usr/lib/x86_64-linux-gnu -name "libnccl.so" 2>/dev/null | head -n 1)"
        else 
            echo "Refer to NCCL documentation for downloads"
        fi
    fi

else 
    echo "NCCL path: $NCCL_PATH"
fi


MPI_PATH=$(find /usr /usr/lib -name "libmpi.so" 2>/dev/null | head -n 1)

if [ MPI_PATH == "" ]; then
    echo "MPI library not found."
    if [ "$(nvcc --version | sed -n 's/.*release \([^,]*\),.*/\1/p') | cut -d. -f1" -lt 12 ]; then
        echo "CUDA VERSION NOT SUFFICIENT"
    
    else
        echo "Download MPI: [y/n]" 
        read mpi_download_allow
        if [ "$mpi_download_allow" == "y" ]; then
            # sudo apt update
            sudo apt update
            sudo apt install openmpi-bin libopenmpi-dev
            echo "MPI_PATH after download: $(find /usr /usr/lib -name "libmpi.so")"
        else 
            echo "Refer to MPI documentation for downloads"
        fi
    fi

else 
    echo "MPI path: $MPI_PATH"
fi










