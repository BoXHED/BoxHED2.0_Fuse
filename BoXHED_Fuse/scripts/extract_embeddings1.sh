cd ../src

python extract_embeddings1.py --test --gpu-no 7 --ckpt-dir /home/ugrads/a/aa_ron_su/BoXHED_Fuse/BoXHED_Fuse/model_outputs/testing/Clinical-T5-Base_TARGET_2_rad_recent_out/1/results --ckpt-model-name model_checkpoint_epoch1.pt --note-type radiology --model-name Clinical-T5-Base --model-type T5 --noteid-mode recent --target 2
