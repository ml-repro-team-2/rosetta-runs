gan=styleganxl

batch_size=10
epochs=160
classidx=550

discrs=("resnet50" "clip" "dino" "dino_vitb8" "dino_vitb16")

for discrmode in "${discrs[@]}";
do
    python match.py --device cuda --save_path matches/$gan/$discrmode/$classidx --gan_mode $gan --discr_mode $discrmode --batch_size $batch_size --epochs $epochs --class $classidx
done