cd $BHF_ROOT

find . -type d -name 'testing' -exec du -sh {} +
# This first command finds all testing files, listing the data size. 
# Example output:
# 33G     ./BoXHED_Fuse/model_outputs/testing
# 3.7M    ./BoXHED_Fuse/JSS_SUBMISSION_NEW/data/testing
# 39M     ./BoXHED_Fuse/JSS_SUBMISSION_NEW/data/final/testing
# 3.4M    ./BoXHED_Fuse/JSS_SUBMISSION_NEW/data/final/Clinical-T5-Base_rad_test_out/1/testing
# 3.6M    ./BoXHED_Fuse/JSS_SUBMISSION_NEW/data/final/Clinical-T5-Base_rad_out/1/testing
# 420K    ./BoXHED_Fuse/JSS_SUBMISSION_NEW/data/embs/testing
# 116K    ./BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/testing

# This second command deletes all testing directories
read -p "Are you sure you want to delete all testing directories? (Y/[N]) " answer
if [ "$answer" = "Y" ] || [ "$answer" = "y" ]; then
    find . -type d -name 'testing' -exec rm -r {} +
else
    echo "Deletion cancelled."
fi


