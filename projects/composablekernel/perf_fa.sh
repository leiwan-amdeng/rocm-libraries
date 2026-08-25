for ((i=0; i<1000; i++)); do
    echo "这是第 $((i+1)) 次执行"
    # 在这里写你的命令，比如跑测试
    ./build-gfx1201-fmha-nosplit115-simplify-profdebug/bin/tile_example_fmha_fwd \
    -prec=bf16 \
    -d=128 \
    -d_v=128 \
    -b=1 \
    -h=30 \
    -s=4224 \
    -mask=0 \
    -lse=0 \
    -p_drop=0 \
    -vlayout=r \
    -kname=1 \
    -warmup=110 \
    -repeat=10
done