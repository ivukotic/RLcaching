LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH
PATH=/usr/local/cuda-10.2/bin:/usr/local/bin:/usr/bin:/bin
export PATH
echo "========= all set up. ============"
ls
"$@"