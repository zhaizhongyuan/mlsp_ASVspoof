# echo "testing with dev data on features: mfcc"
# python3 gmm_test.py --data_path ./data/dev/mfcc.pkl \
#                     --model_path_bon ./model/mfcc/bon.gmm \
#                     --model_path_sp ./model/mfcc/sp.gmm

# echo "testing with test data on features: mfcc"
# python3 gmm_test.py --data_path ./data/test/mfcc.pkl \
#                     --model_path_bon ./model/mfcc/bon.gmm \
#                     --model_path_sp ./model/mfcc/sp.gmm

python gmm_test_output.py --data_path ./data/eval/eval-mfcc.pkl \
                         --model_path_bon ./model/mfcc/bon.gmm \
                         --model_path_sp ./model/mfcc/sp.gmm \
                         --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                         --output_path ./output/mfcc-eval.text

python score_eval.py --cm_score ./output/mfcc-eval.text \
                    --asv_score ./tDCF_python/scores/ASVspoof2019_LA_eval_asv_scores.txt