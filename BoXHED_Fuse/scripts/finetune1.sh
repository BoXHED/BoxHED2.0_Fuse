cd ../src

python finetune1.py --test --gpu-no 7 --target 2 --note-type radiology --model-name Clinical-T5-Base --model-type T5 --num-epochs 1 --noteid-mode recent --target 2
